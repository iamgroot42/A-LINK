import load_data
import committee
import model
import noise

import numpy as np

# Global 
HIGHRES = (224, 224)
DATARES = (50, 50)
LOWRES = (32, 32)
POOLRATIO = 0.75
BIGRATIO = 0.5
N_CLASSES = None
BATCH_SIZE = 16
ACTIVE_COUNT = 0


def init_data(data_dir):
	global LOWRES, POOLRATIO, BIGRATIO, N_CLASSES
	X, Y = load_data.construct_data(data_dir)
	N_CLASSES = np.max(Y) + 1
	return load_data.split_into_sets(X, Y, lowres=LOWRES, pool_ratio=POOLRATIO, big_ratio=BIGRATIO)


def one_hot(Y):
	global N_CLASSES
	y_ = np.zeros((len(Y), N_CLASSES))
	y_[np.arange(len(Y)), Y] = 1
	return y_


if __name__ == "__main__":
	import sys
	(unl_x, unl_y), (x_hr,y_hr), (x_lr,y_lr) = init_data(sys.argv[1])
	print('Loaded data')
	x_highres = load_data.resize(x_hr, HIGHRES)

	#ensemble = [model.FaceVGG16(HIGHRES, N_CLASSES, 512), model.RESNET50(HIGHRES, N_CLASSES)]
	ensemble = [model.RESNET50(HIGHRES, N_CLASSES)]
	ensembleNoise = [noise.Gaussian() for _ in ensemble]

	# Finetune high-resolution models
	for individualModel in ensemble:
		individualModel.finetune(x_highres, one_hot(y_hr), 1, 16)
	print('Finetuned high-resolution models')

	# Train low-resolution model
	lowResModel = model.SmallRes(LOWRES, N_CLASSES)
	lowResModel.finetune(x_lr, one_hot(y_lr), 1, 16)
	print('Finetuned low resolution model')

	# Ready committee of models
	bag = committee.Bagging(N_CLASSES, ensemble, ensembleNoise)
	lowresModel = model.SmallRes(LOWRES, N_CLASSES)

	for i in range(0, len(unl_x), BATCH_SIZE):
		batch_x = unl_x[i * BATCH_SIZE: (i+1)* BATCH_SIZE]
		highres_batch_x = load_data.resize(batch_x, HIGHRES)

		# Get predictions made by committee
		ensemblePredictions = bag.predict(highres_batch_x)

		# Get images with added noise
		noisy_data = bag.attackModel(highres_batch_x, LOWRES)

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
			if np.argmax(ensemblePredictions[j]) == unl_y[i * BATCH_SIZE: (i+1)* BATCH_SIZE][j]:
				queryIndices.append(j)

		# Update count of queries to oracle
		ACTIVE_COUNT += len(queryIndices)

		# Finetune low res model with this actively selected data points
		lowResModel.finetune(batch_x[queryIndices], ensemblePredictions[queryIndices], 1, 16)
