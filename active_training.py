import load_data
import itertools
import committee
import model
import noise
import helpers

import numpy as np
import tensorflow as tf

import keras

from keras_vggface import utils
from tensorflow.python.platform import flags
from sklearn.metrics import confusion_matrix

# Set seed for reproducability
tf.set_random_seed(42)

# Don't hog GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
keras.backend.set_session(sess)

# Global
HIGHRES = (224, 224)
DATARES = (150, 150)
LOWRES = (32, 32)
ACTIVE_COUNT = 0

FLAGS = flags.FLAGS

flags.DEFINE_string('imagesDir', 'data/', 'Path to all images')
flags.DEFINE_string('lowResImagesDir', 'data_final/lowres/', 'Path to low-res images')
flags.DEFINE_string('highResImagesDir', 'data_final/highres/', 'Path to high-res images')
flags.DEFINE_string('unlabelledList', 'fileLists/unlabelledData.txt', 'Path to unlabelled images list')
flags.DEFINE_string('testDataList', 'fileLists/testData.txt', 'Path to test images list')
flags.DEFINE_integer('batch_size', 16, 'Batch size while sampling from unlabelled data')
flags.DEFINE_integer('low_epochs', 50, 'Number of epochs while training low resolution model')
flags.DEFINE_integer('high_epochs', 20, 'Number of epochs while fine-tuning high resolution model')
flags.DEFINE_integer('batch_send', 64, 'Batch size while finetuning low-resolution model')
flags.DEFINE_float('active_ratio', 1.0, 'Upper cap on ratio of unlabelled examples to be qurried for labels')


if __name__ == "__main__":
	# Initialize generator for unlabelled data
	unlabelledImagesGenerator = load_data.getUnlabelledData(FLAGS.imagesDir, FLAGS.unlabelledList, FLAGS.batch_size)

	# Initialize generator for high-resolution data
	highgenTrain, highgenVal = load_data.returnGenerators(FLAGS.highResImagesDir + "train", FLAGS.highResImagesDir + "val", HIGHRES, 16, helpers.hr_preprocess)
	
	# Load low-resolution data
	(X_low_train, Y_low_train), (X_low_val, Y_low_val), lowMap = load_data.resizeLoadDataAll(FLAGS.imagesDir, FLAGS.lowResImagesDir + "train", FLAGS.lowResImagesDir + "val", LOWRES) 

	#ensemble = [model.FaceVGG16(HIGHRES, N_CLASSES, 512), model.RESNET50(HIGHRES, N_CLASSES)]
	ensemble = [model.RESNET50(HIGHRES, N_CLASSES)]
	ensembleNoise = [noise.Gaussian() for _ in ensemble]
	#ensembleNoise = [noise.Noise() for _ in ensemble]

	# Ready committee of models
	bag = committee.Bagging(N_CLASSES, ensemble, ensembleNoise)
	lowResModel = model.SmallRes(LOWRES, N_CLASSES)
	
	# Train low-resolution model
	if not lowResModel.maybeLoadFromMemory():
	        lowResModel.trainModel(X_low_train, Y_low_train, X_low_val, Y_low_val, FLAGS.low_epochs, 16, 1)
        	print('Trained low resolution model')
		lowResModel.save()
	else:
		print("Loaded", lowResModel.modelName, "from memory")

	# Finetune high-resolution model(s), if not already trained
	for individualModel in ensemble:
		if not individualModel.maybeLoadFromMemory():
			individualModel.finetuneGenerator(highgenTrain, highgenVal, 2000, 16, FLAGS.high_epochs, 1)
			individualModel.save()
		else:
			print("Loaded", individualModel.modelName, "from memory")
	print('Finetuned high-resolution models')
	
	# Train low res model only when batch length crosses threshold
	train_lr_x = np.array([])
	train_lr_y = np.array([])
	UN_SIZE = load_data.getContentsSize(FLAGS.testDataList)

	# Cumulative data (don't forget old data)
	cumu_x = X_low_train
	cumu_y = Y_low_train

	while True:
		try:
			batch_x, batch_y = unlabelledImagesGenerator.next()
		except:
			break

		batch_x_hr = load_data.resize(batch_x, HIGHRES)
		batch_x_lr = load_data.resize(batch_x, LOWRES)

		# Get predictions made by committee
		ensemblePredictions = bag.predict(batch_x_hr)

		# Get images with added noise
		noisy_data = bag.attackModel(batch_x_hr, LOWRES)

		# Pass these to low res model, get predictions
		lowResPredictions = lowResModel.predict(noisy_data)

		# Pick examples that were misclassified
		misclassifiedIndices = []
		for j in range(len(lowResPredictions)):
			if lowMapinv[np.argmax(lowResPredictions[j])] != highMapinv[np.argmax(ensemblePredictions[j])]:
				misclassifiedIndices.append(j)

		# Query oracle, pick examples for which ensemble was right
		queryIndices = []
		for j in misclassifiedIndices:
			if np.argmax(ensemblePredictions[j]) == highMap[batch_y[j]]:
				queryIndices.append(j)

		# Update count of queries to oracle
		ACTIVE_COUNT += len(queryIndices)

		# If nothing matches, proceed to next set of predictions
		if len(queryIndices) == 0:
			continue

		# Convert class mappings (high/res)
		intermediate = []
		for qi in queryIndices:
			intermediate.append(lowMap[highMapinv[np.argmax(ensemblePredictions[qi])]])
		intermediate = np.array(intermediate)

		# Gather data to be sent to low res model for training
		if train_lr_x.shape[0] > 0:
			train_lr_x = np.concatenate((train_lr_x, noisy_data[queryIndices]))
			train_lr_y = np.concatenate((train_lr_y, helpers.one_hot(intermediate, N_CLASSES)))
		else:
			train_lr_x = np.copy(noisy_data[queryIndices])
			train_lr_y = np.copy(helpers.one_hot(intermediate, N_CLASSES))

		if train_lr_x.shape[0] >= FLAGS.batch_send:
			# Finetune low res model with this actively selected data points
			# Also add unperturbed versions of these examplesto avoid overfitting on noise
			train_lr_x = np.concatenate((train_lr_x, batch_x_lr[queryIndices]))
			train_lr_y = np.concatenate((train_lr_y, helpers.one_hot(intermediate, N_CLASSES)))
			# Use a lower learning rate for finetuning (accumulate data as a countermeasure against catastrophic forgetting)
			cumu_x = np.concatenate((cumu_x, train_lr_x))
			cumu_y = np.concatenate((cumu_y, train_lr_y))
			lowResModel.finetuneDenseOnly(cumu_x, cumu_y, 3, 16, 0)
			train_lr_x = np.array([])
			train_lr_y = np.array([])
			# Log test accuracy
			testGen = load_data.testDataGenerator(FLAGS.imagesDir, FLAGS.testDataList, HIGHRES, LOWRES, 128)
			print('Test accuracy after this pass:', helpers.calculate_accuracy(testGen, lowResModel, "low", lowMap, 1))

		# Stop algorithm if limit reached/exceeded
		if int(FLAGS.active_ratio * UN_SIZE) <= ACTIVE_COUNT:
			break

	# Print count of images queried so far
	print("Active Count:", ACTIVE_COUNT, "out of:", UN_SIZE)

	# Log statistics
	testGen = load_data.testDataGenerator(FLAGS.imagesDir, FLAGS.testDataList, HIGHRES, LOWRES, 128)
	print('High-res model test accuracy:', helpers.calculate_accuracy(testGen, bag, "high", highMap, 1))
