# For headless machines
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import load_data
import itertools
import committee
import model
import noise

import numpy as np
import tensorflow as tf
import keras

from keras_vggface import utils
from sklearn.metrics import confusion_matrix


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
LOWRESIMAGESDIR = "data_final/lowres" # Path to low-res images
HIGHRESIMAGESDIR = "data_final/highres"# Path to high-res images
UNLABALLEDLIST = "fileLists/unlabelledData.txt" # Path to unlabelled images list
TESTDATALIST = "fileLists/testData.txt"  # Path to test images list

# Don't hog GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
keras.backend.set_session(sess)

def one_hot(Y):
	global N_CLASSES
	y_ = np.zeros((len(Y), N_CLASSES))
	y_[np.arange(len(Y)), Y] = 1
	return y_


def hr_preprocess(X):
	X_temp = np.copy(X)
	return utils.preprocess_input(X_temp, version=1)


def lr_preprocess(X):
	return X / 255.0


def get_transformed_predictions(Y_true, mapping):
	ret = []
	for y in Y_true:
		ret.append(mapping[y])
	return np.array(ret)


def calculate_accuracy(Y_pred, Y_labels, mapping):
	score = 0.0
	for i in range(len(Y_pred)):
		if Y_pred[i] == mapping[Y_labels[i]]:
			score += 1.0
	return score / len(Y_pred)


def calculate_topNaccuracy(Y_pred, Y_labels, mapping, N=5):
	score = 0.0
	for i in range(len(Y_pred)):
		sorted_preds = np.argsort(-Y_pred[i])
		for j in range(N):
			if mapping[Y_labels[i]] == sorted_preds[j]:
				score += 1.0
	return score / len(Y_pred)


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == "__main__":

	X_test_lr, X_test_hr, Y_test = load_data.loadTestData(IMAGESDIR, TESTDATALIST, HIGHRES, LOWRES)
	print('Loaded test data')

	unlabelledImagesGenerator = load_data.getUnlabelledData(IMAGESDIR, UNLABALLEDLIST, BATCH_SIZE)

	lowgenTrain, lowgenVal = load_data.returnGenerators(LOWRESIMAGESDIR + "/train", LOWRESIMAGESDIR + "/val", LOWRES, 16, lr_preprocess)
	highgenTrain, highgenVal = load_data.returnGenerators(HIGHRESIMAGESDIR + "/train", HIGHRESIMAGESDIR + "/val", HIGHRES, 16, hr_preprocess)

	# Get mappings from classnames to softmax indices
	lowMap = lowgenVal.class_indices
	highMap = highgenVal.class_indices

	#ensemble = [model.FaceVGG16(HIGHRES, N_CLASSES, 512), model.RESNET50(HIGHRES, N_CLASSES)]
	ensemble = [model.RESNET50(HIGHRES, N_CLASSES)]
	ensembleNoise = [noise.Gaussian() for _ in ensemble]

	# Finetune high-resolution models
	for individualModel in ensemble:
		individualModel.finetuneGenerator(highgenTrain, highgenVal, 2000, 16, 3, 0)
	print('Finetuned high-resolution models')

	# Train low-resolution model
	lowResModel = model.SmallRes(LOWRES, N_CLASSES)
	lowResModel.finetuneGenerator(lowgenTrain, lowgenVal, 2000, 16, 50, 0)
	print('Finetuned low resolution model')

	# Calculate accuracy of low-res model at this stage
	lowresPreds = lowResModel.predict(X_test_lr)
	print('Low-res model test accuracy:', calculate_accuracy(np.argmax(lowresPreds,axis=1), Y_test, lowMap))	
	print('Low-res model top-5 accuracy:', calculate_topNaccuracy(lowresPreds, Y_test, lowMap, 5))	

	# Ready committee of models
	bag = committee.Bagging(N_CLASSES, ensemble, ensembleNoise)
	lowresModel = model.SmallRes(LOWRES, N_CLASSES)

	# Train low res model only when batch length crosses threshold
	train_lr_x = np.array([])
	train_lr_y = np.array([])
	BATCH_SEND = 32
	UN_SIZE = 25117

	for i in range(0, UN_SIZE, BATCH_SIZE):

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
			lowResModel.finetune(train_lr_x, train_lr_y, 1, 16, 0)
			train_lr_x = np.array([])
			train_lr_y = np.array([])

	# Print count of images queried so far
	print("Active Count:", ACTIVE_COUNT, "out of:", UN_SIZE)

	lowresPreds = lowResModel.predict(X_test_lr)
	highresPreds = bag.predict(X_test_hr)
	numAgree = int(np.sum((np.argmax(lowresPreds,axis=1) == np.argmax(highresPreds,axis=1)) * 1.0))

	# Log accuracies
	print(numAgree, 'out of', len(lowresPreds), 'agree')
	print('Low-res model test accuracy:', calculate_accuracy(np.argmax(lowresPreds,axis=1), Y_test, lowMap))
	print('High-res model test accuracy:', calculate_accuracy(np.argmax(highresPreds,axis=1), Y_test, highMap))

	# Log top-5 accuracies:
	print('Low-res model top-5 accuracy:', calculate_topNaccuracy(lowresPreds, Y_test, lowMap, 5))
	print('High-res model top-5 accuracy:', calculate_topNaccuracy(highresPreds, Y_test, highMap, 5))

	# Save low-res model
	lowResModel.model.save("low_res_model")

	# Confusion matrices
        lr_cnf_matrix = confusion_matrix(get_transformed_predictions(Y_test, lowMap), np.argmax(lowresPreds, axis=1))
        hr_cnf_matrix = confusion_matrix(get_transformed_predictions(Y_test, highMap), np.argmax(highresPreds, axis=1))
        plt.figure()
        plot_confusion_matrix(lr_cnf_matrix, classes=range(N_CLASSES), title='Low resolution model')
        plt.savefig('low_res_confmat.png')
        plt.figure()
        plot_confusion_matrix(hr_cnf_matrix, classes=range(N_CLASSES), title='High resolution model')
        plt.savefig('high_res_confmat.png')
