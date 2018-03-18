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
FEATURERES = (2048,)
FRAMEWORK_BS = 16
N_CLASSES = 2
ACTIVE_COUNT = 0

FLAGS = flags.FLAGS

flags.DEFINE_string('dataDirPrefix', 'DFW/DFW_Data/', 'Path to DFW data directory')
flags.DEFINE_string('trainImagesDir', 'Training_data', 'Path to DFW training-data images')
flags.DEFINE_string('testImagesDir', 'Testing_data', 'Path to DFW testing-data imgaes')
flags.DEFINE_integer('batch_size', 16, 'Batch size while sampling from unlabelled data')
flags.DEFINE_integer('dig_epochs', 50, 'Number of epochs while training disguised-faces model')
flags.DEFINE_integer('undig_epochs', 20, 'Number of epochs while fine-tuning undisguised-faces model')
flags.DEFINE_integer('batch_send', 64, 'Batch size while finetuning disguised-faces model')
flags.DEFINE_float('active_ratio', 1.0, 'Upper cap on ratio of unlabelled examples to be qurried for labels')


if __name__ == "__main__":
	# Define image featurization model
	conversionModel = siamese.RESNET50(IMAGERES)

	# Load images, convert to feature vectors for faster processing
	(X_plain, X_dig, X_imp) = readDFW.getAllTrainData(FLAGS.dataDirPrefix, FLAGS.trainImagesDir, IMAGERES, conversionModel)
	(X_plain_raw, X_dig_raw) = readDFW.getRawTrainData(FLAGS.dataDirPrefix, FLAGS.trainImagesDir, IMAGERES)

	# Split X_dig person-wise for pretraining & framework data
	(X_dig_pre, _) = readDFW.splitDisguiseData(X_dig, pre_ratio=0.5)
	(_, X_dig_post) = readDFW.splitDisguiseData(X_dig_raw, pre_ratio=0.5)

	ensemble = [siamese.SiameseNetwork(FEATURERES, "ensemble1", 0.1)]
	# ensembleNoise = [noise.Gaussian() for _ in ensemble]
	ensembleNoise = [noise.Noise() for _ in ensemble]

	# Ready committee of models
	bag = committee.Bagging(ensemble, ensembleNoise)
	disguisedFacesModel = siamese.SiameseNetwork(FEATURERES, "disguisedModel", 0.1)
	
	# Create generators
	normGen = readDFW.getNormalGenerator(X_plain, FLAGS.batch_size)
	disgGen = readDFW.getNormalGenerator(X_dig_pre, FLAGS.batch_size)
	normImpGen = readDFW.getNormalGenerator(X_imp, FLAGS.batch_size)
	impGenNorm  = readDFW.getImposterGenerator(X_plain, X_imp, FLAGS.batch_size)
	impGenDisg = readDFW.getImposterGenerator(X_dig_pre, X_imp, FLAGS.batch_size)


	dataGen = readDFW.getGenerator(disgGen, normImpGen, impGenDisg, FLAGS.batch_size, 0)
	disguisedFacesModel.customTrainModel(dataGen, FLAGS.dig_epochs, FLAGS.batch_size, 0.2)
	exit()

	# Finetune disguised-faces model
	if not disguisedFacesModel.maybeLoadFromMemory():
		trainDatagen = readDFW.getGenerator(disgGen, normImpGen, impGenDisg, FLAGS.batch_size, 0)
		valDatagen = None
		disguisedFacesModel.trainModel(trainDatagen, valDatagen, FLAGS.dig_epochs, FLAGS.batch_size, verbose=1)
		disguisedFacesModel.save()
		print('Finetuned disguised-faces model')
	else:
		print('Loaded disguised-faces model from memory')

	# Finetune undisguised model(s), if not already trained
	for individualModel in ensemble:
		if not individualModel.maybeLoadFromMemory():
			trainDatagen = readDFW.getGenerator(normGen, normImpGen, impGenNorm, FLAGS.batch_size, 0)
			valDatagen = None
			individualModel.trainModel(trainDatagen, valDatagen, FLAGS.undig_epochs, FLAGS.batch_size, verbose=1)
			individualModel.save()
	print('Finetuned undisguised-faces models')

	# Train disguised-faces model only when batch length crosses threshold
	train_df_x = np.array([])
	train_df_y = np.array([])

	# Do something about catastrophic forgetting (?)
	UN_SIZE = 0

	# Framework begins
	print("Framework begins")
	for ii in range(0, len(X_dig_post), FRAMEWORK_BS):
		plain_part = X_plain_raw[ii: ii + FRAMEWORK_BS]
		disguise_part = X_dig_post[ii: ii + FRAMEWORK_BS]

		# Create pairs of images
		batch_x, batch_y = readDFW.createMiniBatch(plain_part, disguise_part)
		UN_SIZE += len(batch_x[0])

		# Get featurized faces to be passed to committee
		batch_x_features = [ conversionModel.process(p) for p in batch_x]

		# Get predictions made by committee
		ensemblePredictions = bag.predict(batch_x_features)

		# Get images with added noise
		noisy_data = bag.attackModel(batch_x, IMAGERES)

		# Get features back from noisy images
                noisy_data = [ conversionModel.process(p) for p in noisy_data ]

		# Pass these to disguised-faces model, get predictions
		disguisedPredictions = disguisedFacesModel.predict(noisy_data)

		# Pick examples that were misclassified
		misclassifiedIndices = []
		for j in range(len(disguisedPredictions)):
			if np.argmax(disguisedPredictions[j]) != np.argmax(ensemblePredictions[j]):
				misclassifiedIndices.append(j)

		# Query oracle, pick examples for which ensemble was right
		queryIndices = []
		for j in misclassifiedIndices:
			if np.argmax(ensemblePredictions[j]) == np.argmax(batch_y[j]):
				queryIndices.append(j)

		# Update count of queries to oracle
		ACTIVE_COUNT += len(queryIndices)

		# If nothing matches, proceed to next set of predictions
		if len(queryIndices) == 0:
			continue

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
			# Also add unperturbed images to avoid overfitting on noise
			train_df_x = np.concatenate((train_df_x, batch_x_features[queryIndices]))
			train_df_y = np.concatenate((train_df_y, helpers.one_hot(intermediate, N_CLASSES)))
			# Use a lower learning rate for finetuning
			disguisedFacesModel.finetune(train_df_x, train_df_y, 3, 16, 0)
			train_df_x = np.array([])
			train_df_y = np.array([])

		# Stop algorithm if limit reached/exceeded
		if int(FLAGS.active_ratio * len(X_dig_post)) <= ACTIVE_COUNT:
			break

	# Print count of images queried so far
	print("Active Count:", ACTIVE_COUNT, "out of:", UN_SIZE)

