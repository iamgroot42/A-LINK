import readDFW
import itertools
import committee
import siamese
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
IMAGERES = (224, 224)
N_CLASSES = None
ACTIVE_COUNT = 0

FLAGS = flags.FLAGS

flags.DEFINE_string('dataDirPrefix', 'DFW/DFW_Data/', 'Path to DFW data directory')
flags.DEFINE_string('trainImagesDir', 'Training_data', 'Path to DFW training-data images')
flags.DEFINE_string('testImagesDir', 'Testing_data', 'Path to DFW testing-data imgaes')
flags.DEFINE_integer('batch_size', 16, 'Batch size while sampling from unlabelled data')
flags.DEFINE_integer('low_epochs', 50, 'Number of epochs while training disguised-faces model')
flags.DEFINE_integer('high_epochs', 20, 'Number of epochs while fine-tuning undisguised-faces model')
flags.DEFINE_integer('batch_send', 64, 'Batch size while finetuning disguised-faces model')
flags.DEFINE_float('active_ratio', 1.0, 'Upper cap on ratio of unlabelled examples to be qurried for labels')


if __name__ == "__main__":
	(X_plain, X_dig, X_imp) = readDFW.getAllTrainData(FLAGS.dataDirPrefix, FLAGS.trainImagesDir, IMAGERES)
	(X_test, Y_test) = readDFW.getAllTestData(FLAGS.dataDirPrefix, FLAGS.testImagesDir, IMAGERES)

	#ensemble = [model.FaceVGG16(IMAGERES, N_CLASSES, 512), model.RESNET50(IMAGERES, N_CLASSES)]
	ensemble = [model.RESNET50(IMAGERES, "ensemble1" ,1.0)]
	# ensembleNoise = [noise.Gaussian() for _ in ensemble]
	ensembleNoise = [noise.Noise() for _ in ensemble]

	# Ready committee of models
	bag = committee.Bagging(ensemble, ensembleNoise)
	disguisedFacesModel = model.RESNET50(IMAGERES, "standalone1", 1.0)
	
	# Finetune disguised-faces model
	trainDatagen = readDFW.getGenerator(X_dig, X_imp, FLAGS.batch_size)
	disguisedFacesModel.trainModel(trainDatagen, FLAGS.epochs, FLAGS.batch_size, verbose=1)
	print('Finetuned disguised-faces model')

	# Finetune undisguised model(s), if not already trained
	for individualModel in ensemble:
		# if not individualModel.maybeLoadFromMemory():
		trainDatagen = readDFW.getGenerator(X_plain, X_imp, FLAGS.batch_size)
		individualModel.trainModel(trainDatagen, FLAGS.epochs, FLAGS.batch_size, verbose=1)
	print('Finetuned undisguised-faces models')

	# Calculate accuracy of disguised-faces model at this stage
	print('Disguised model test accuracy:', disguisedFacesModel.model.testAccuracy(X_test, Y_test))
	exit()

	# Train disguised-faces model only when batch length crosses threshold
	train_df_x = np.array([])
	train_df_y = np.array([])
	UN_SIZE = len(X_dig_ft)

	# Cumulative data (don't forget old data)
	cumu_x = X_dig
	cumu_y = Y_dig_train

	for ii in range(0, len(X_dig_ft), 16):
		batch_x, batch_y = X_dig_ft[ii:ii + 16], Y_dig_ft[ii: ii + 16]

		# Get predictions made by committee
		ensemblePredictions = bag.predict(batch_x)

		# Get images with added noise
		noisy_data = bag.attackModel(batch_x, IMAGERES)

		# Pass these to low res model, get predictions
		disguisedPredictions = disguisedFacesModel.predict(noisy_data)

		# Pick examples that were misclassified
		misclassifiedIndices = []
		for j in range(len(disguisedPredictions)):
			if revIndices[np.argmax(disguisedPredictions[j])] != revIndices[np.argmax(ensemblePredictions[j])]:
				misclassifiedIndices.append(j)

		# Query oracle, pick examples for which ensemble was right
		queryIndices = []
		for j in misclassifiedIndices:
			if revIndices[np.argmax(ensemblePredictions[j])] == revIndices[np.argmax(batch_y[j])]:
				queryIndices.append(j)

		# Update count of queries to oracle
		ACTIVE_COUNT += len(queryIndices)

		# If nothing matches, proceed to next set of predictions
		if len(queryIndices) == 0:
			continue

		# Convert class mappings (high/res)
		intermediate = []
		for qi in queryIndices:
			intermediate.append(np.argmax(ensemblePredictions[qi]))
		intermediate = np.array(intermediate)

		# Gather data to be sent to low res model for training
		if train_df_x.shape[0] > 0:
			train_df_x = np.concatenate((train_df_x, noisy_data[queryIndices]))
			train_df_y = np.concatenate((train_df_y, helpers.one_hot(intermediate, N_CLASSES)))
		else:
			train_df_x = np.copy(noisy_data[queryIndices])
			train_df_y = np.copy(helpers.one_hot(intermediate, N_CLASSES))

		if train_df_x.shape[0] >= FLAGS.batch_send:
			# Finetune disguised-faces model with this actively selected data points
			# Also add unperturbed versions of these examples to avoid overfitting on noise
			train_df_x = np.concatenate((train_df_x, batch_x[queryIndices]))
			train_df_y = np.concatenate((train_df_y, helpers.one_hot(intermediate, N_CLASSES)))
			# Use a lower learning rate for finetuning (accumulate data as a countermeasure against catastrophic forgetting)
			cumu_x = np.concatenate((cumu_x, train_df_x))
			cumu_y = np.concatenate((cumu_y, train_df_y))
			disguisedFacesModel.finetuneDenseOnly(cumu_x, cumu_y, 3, 16, 0)
			train_df_x = np.array([])
			train_df_y = np.array([])
			# Log test accuracy
			print('Disguised model test accuracy (after this pass):', disguisedFacesModel.model.evaluate(X_val, Y_val)[1])

		# Stop algorithm if limit reached/exceeded
		if int(FLAGS.active_ratio * UN_SIZE) <= ACTIVE_COUNT:
			break

	# Print count of images queried so far
	print("Active Count:", ACTIVE_COUNT, "out of:", UN_SIZE)
