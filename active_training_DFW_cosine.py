import readDFW_cosine
import itertools
import committee
import siamese_cosine
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
flags.DEFINE_integer('mixture_ratio', 2, 'Ratio of unperturbed:perturbed examples while finetuning network')


if __name__ == "__main__":

	# Load images
	(X_plain, X_dig, X_imp) = readDFW.getAllTrainData(FLAGS.dataDirPrefix, FLAGS.trainImagesDir, IMAGERES)

	# Split X_dig person-wise for pretraining & framework data
	(X_dig_pre, X_dig_post) = readDFW.splitDisguiseData(X_dig, pre_ratio=0.5)

	ensemble = [siamese_cosine.FaceVGG16(FEATURERES, "ensemble1_cosine", 0.1)]
	ensembleNoise = [noise.Gaussian() for _ in ensemble]
	#ensembleNoise = [noise.Noise() for _ in ensemble]

	# Ready committee of models
	bag = committee.Bagging(ensemble, ensembleNoise)
	disguisedFacesModel = siamese.FaceVGG16(FEATURERES, "disguisedModel_cosine", 0.1)
	
	# Create generators
	normGen = readDFW.getNormalGenerator(X_plain, FLAGS.batch_size)
	disgGen = readDFW.getNormalGenerator(X_dig_pre, FLAGS.batch_size)
	normImpGen = readDFW.getNormalGenerator(X_imp, FLAGS.batch_size)
	impGenNorm  = readDFW.getImposterGenerator(X_plain, X_imp, FLAGS.batch_size)
	impGenDisg = readDFW.getImposterGenerator(X_dig_pre, X_imp, FLAGS.batch_size)

	# Finetune disguised-faces model
	if not disguisedFacesModel.maybeLoadFromMemory():
		dataGen = readDFW.getGenerator(disgGen, normImpGen, impGenDisg, FLAGS.batch_size, 0)
		disguisedFacesModel.customTrainModel(dataGen, FLAGS.dig_epochs, FLAGS.batch_size, 0.2)
		disguisedFacesModel.save()
		print('Finetuned disguised-faces model')
	else:
		print('Loaded disguised-faces model from memory')

	# Finetune undisguised model(s), if not already trained
	for individualModel in ensemble:
		if not individualModel.maybeLoadFromMemory():
			dataGen = readDFW.getGenerator(normGen, normImpGen, impGenNorm, FLAGS.batch_size, 0)
			individualModel.customTrainModel(dataGen, FLAGS.undig_epochs, FLAGS.batch_size, 0.2)
			individualModel.save()
	print('Finetuned undisguised-faces models')

	# Train disguised-faces model only when batch length crosses threshold
	train_df_left_x = np.array([])
	train_df_right_x = np.array([])
	train_df_y = np.array([])

	# Do something about catastrophic forgetting (?)
	UN_SIZE = 0

	# Framework begins
	print("Framework begins with a pool of", len(X_dig_post))
	dataGen = readDFW.getGenerator(normGen, normImpGen, impGenNorm, FLAGS.batch_size, 0)
	for ii in range(0, len(X_dig_post), FRAMEWORK_BS):
		print( (ii / FRAMEWORK_BS) + 1, "iteration")
		plain_part = X_plain[ii: ii + FRAMEWORK_BS]
		disguise_part = X_dig_post[ii: ii + FRAMEWORK_BS]

		# Create pairs of images
		batch_x, batch_y = readDFW.createMiniBatch(plain_part, disguise_part)
		UN_SIZE += len(batch_x[0])

		# Get predictions made by committee
		ensemblePredictions = bag.predict(batch_x)

		# Get images with added noise
		noisy_data = [ bag.attackModel(p, IMAGERES) for p in batch_x]

		# Pass these to disguised-faces model, get predictions
		disguisedPredictions = disguisedFacesModel.predict(noisy_data)

		# Pick examples that were misclassified
		misclassifiedIndices = []
		for j in range(len(disguisedPredictions)):
			c1 = disguisedPredictions[j][0] >= 0.5
			c2 = ensemblePredictions[j][0] >= 0.5
			if c1 != c2:
				misclassifiedIndices.append(j)

		# Query oracle, pick examples for which ensemble was right
		queryIndices = []
		for j in misclassifiedIndices:
			c1 = ensemblePredictions[j][0] >= 0.5
			c2 = batch_y[j][0] >= 0.5
			if c1 == c2:
				queryIndices.append(j)

		# Update count of queries to oracle
		ACTIVE_COUNT += len(queryIndices)
		print("Active Count so far", ACTIVE_COUNT)

		# If nothing matches, proceed to next set of predictions
		if len(queryIndices) == 0:
			continue

		intermediate = []
		for i in queryIndices:
			intermediate.append(ensemblePredictions[i][0])
		intermediate = np.array(intermediate)

		# Gather data to be sent to low res model for training
		if train_df_y.shape[0] > 0:
			train_df_left_x = np.concatenate((train_df_left_x, noisy_data[0][queryIndices]))
			train_df_right_x = np.concatenate((train_df_right_x, noisy_data[1][queryIndices]))
			train_df_y = np.concatenate((train_df_y, helpers.roundoff(intermediate)))
		else:
			train_df_left_x = np.copy(noisy_data[0][queryIndices])
			train_df_right_x = np.copy(noisy_data[1][queryIndices])
			train_df_y = np.copy(helpers.roundoff(intermediate))

		if train_df_y.shape[0] >= FLAGS.batch_send:
			# Finetune disguised-faces model with this actively selected data points
			# Also add unperturbed images to avoid overfitting on noise
			(X_old_left, X_old_right), Y_old = dataGen.next()
			for _ in range(FLAGS.mixture_ratio - 1):
				X_old_temp, Y_old_temp = dataGen.next()
				X_old_left = np.concatenate((X_old_left, X_old_temp[0]))
				X_old_right = np.concatenate((X_old_right, X_old_temp[1]))
				Y_old = np.concatenate((Y_old, Y_old_temp))
			train_df_left_x = np.concatenate((train_df_left_x, batch_x[0][queryIndices], X_old_left))
			train_df_right_x = np.concatenate((train_df_right_x, batch_x[1][queryIndices], X_old_right))
			train_df_y = np.concatenate((train_df_y, helpers.roundoff(intermediate), Y_old))
			# Use a lower learning rate for finetuning ?
			disguisedFacesModel.finetune([train_df_left_x, train_df_right_x], train_df_y, 3, 16, 1)
			train_df_left_x = np.array([])
			train_df_right_x = np.array([])
			train_df_y = np.array([])

		# Stop algorithm if limit reached/exceeded
		if int((FLAGS.active_ratio * len(X_dig_post) * (FRAMEWORK_BS - 1)) / 2) <= ACTIVE_COUNT:
			break

	# Print count of images queried so far
	print("Active Count:", ACTIVE_COUNT, "out of:", UN_SIZE)

	# Save retrained model
	disguisedFacesModel.save("fineTuned")
