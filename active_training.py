import load_data
import committee
import model
import noise

import numpy as np

# Global
HIGHRES = (224, 224)
DATARES = (50, 50)
LOWRES = (32, 32)
POOLRATIO = 0.6
BIGRATIO = 0.5
N_CLASSES = 337
BATCH_SIZE = 16
ACTIVE_COUNT = 0

# Image directories
IMAGESDIR = "data/" # Path to all images
LOWRESIMAGESDIR = "dataFinal/lowres" # Path to low-res images
HIGHRESIMAGESDIR = "dataFinal/highres"# Path to high-res images
UNLABALLEDLIST = "unlabelledData.txt" # Path to unlabelled images list
TESTDATALIST = "testData.txt"  # Path to test images list


def one_hot(Y):
	global N_CLASSES
	y_ = np.zeros((len(Y), N_CLASSES))
	y_[np.arange(len(Y)), Y] = 1
	return y_


if __name__ == "__main__":
	import sys

	X_test_lr, X_test_hr, Y_test = load_data.loadTestData(IMAGESDIR, TESTDATALIST, HIGHRES, LOWRES)
	print('Loaded test data')

	unlabelledImagesGenerator = load_data.getUnlabelledData(IMAGESDIR, UNLABALLEDLIST, BATCH_SIZE)

	lowgenTrain, lowgenVal = load_data.returnGenerators(LOWRESIMAGESDIR + "/train", LOWRESIMAGESDIR + "/val", LOWRES, 16)
	highgenTrain, highgenVal = load_data.returnGenerators(HIGHRESIMAGESDIR + "/train", HIGHRESIMAGESDIR + "/val", HIGHRES, 16)

	#ensemble = [model.FaceVGG16(HIGHRES, N_CLASSES, 512), model.RESNET50(HIGHRES, N_CLASSES)]
	ensemble = [model.RESNET50(HIGHRES, N_CLASSES)]
	ensembleNoise = [noise.Gaussian() for _ in ensemble]

	# Finetune high-resolution models
	for individualModel in ensemble:
		individualModel.finetuneGenerator(highgenTrain, highgenVal, 2000, 16)
	print('Finetuned high-resolution models')

	# Train low-resolution model
	lowResModel = model.SmallRes(LOWRES, N_CLASSES)
	lowResModel.finetuneGenerator(lowgenTrain, lowgenVal, 2000, 16)
	print('Finetuned low resolution model')

	# Ready committee of models
	bag = committee.Bagging(N_CLASSES, ensemble, ensembleNoise)
	lowresModel = model.SmallRes(LOWRES, N_CLASSES)

	# Train low res model only when batch length crosses threshold
	train_lr_x = np.array([])
	train_lr_y = np.array([])
	BATCH_SEND = 16

	for i in range(0, len(unl_x), BATCH_SIZE):
		batch_x_lr, batch_x_hr, batch_y = unlabelledImagesGenerator.next()

		if highres_batch_x.shape[0] == 0:
			break

		# Get predictions made by committee
		ensemblePredictions = bag.predict(batch_x_hr)

		# Get images with added noise
		noisy_data = bag.attackModel(batch_x_hr, LOWRES)

		# Pass these to low res model, get predictions
		lowResPredictions = lowresModel.predict(noisy_data)

		# Pick examples that were misclassified
		misclassifiedIndices = []

		for j in range(len(lowResPredictions)):
			if np.argmax(lowResPredictions[j]) != np.argmax(ensemblePredictions[j]):
				misclassifiedIndices.append(j)

		# Query oracle, pick examples for which ensemble was right
		queryIndices = []
		for j in misclassifiedIndices:
			if np.argmax(ensemblePredictions[j]) == batch_y[j]:
				queryIndices.append(j)

		# If nothing matches, proceed to next set of predictions
		if len(queryIndices) == 0:
			continue

		# Update count of queries to oracle
		ACTIVE_COUNT += len(queryIndices)

		# Gather data to be sent to low res model for training
		if train_lr_x.shape[0] > 0:
			train_lr_x = np.concatenate((train_lr_x, noisy_data[queryIndices]))
			train_lr_y = np.concatenate((train_lr_y, one_hot(np.argmax(ensemblePredictions[queryIndices], axis=1))))
		else:
			train_lr_x = noisy_data[queryIndices]
			train_lr_y = one_hot(np.argmax(ensemblePredictions[queryIndices], axis=1))

		if train_lr_x.shape[0] >= BATCH_SEND:
			# Finetune low res model with this actively selected data points
			lowResModel.finetune(train_lr_x, train_lr_y, 1, 16)
			train_lr_x = np.array([])
			train_lr_y = np.array([])
			print "Trained low-res model again!"

		# Print count of images queried so far
		print("Active Count:", ACTIVE_COUNT)
