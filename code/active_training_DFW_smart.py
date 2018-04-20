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
#FEATURERES = (25088,)
FRAMEWORK_BS = 16
ACTIVE_COUNT = 0

FLAGS = flags.FLAGS

flags.DEFINE_string('dataDirPrefix', 'DFW/DFW_Data/', 'Path to DFW data directory')
flags.DEFINE_string('trainImagesDir', 'Training_data', 'Path to DFW training-data images')
flags.DEFINE_string('testImagesDir', 'Testing_data', 'Path to DFW testing-data imgaes')
flags.DEFINE_integer('ft_epochs', 3, 'Number of epochs while finetuning model')
flags.DEFINE_integer('batch_size', 16, 'Batch size while sampling from unlabelled data')
flags.DEFINE_integer('dig_epochs', 2, 'Number of epochs while training disguised-faces model')
flags.DEFINE_integer('undig_epochs', 2, 'Number of epochs while fine-tuning undisguised-faces model')
flags.DEFINE_integer('batch_send', 64, 'Batch size while finetuning disguised-faces model')
flags.DEFINE_integer('mixture_ratio', 2, 'Ratio of unperturbed:perturbed examples while finetuning network')
flags.DEFINE_string('out_model', 'models/fineTuned', 'Name of model to be saved after finetuning')
flags.DEFINE_boolean('refine_models', False, 'Refine previously trained models?')
flags.DEFINE_boolean('augment', False, 'Augmente data while finetuning covariate-based model?')


if __name__ == "__main__":
	# Define image featurization model
	conversionModel = siamese.RESNET50(IMAGERES)
	#conversionModel = siamese.FaceVGG16(IMAGERES)

	# Load images, convert to feature vectors for faster processing
	(X_plain, X_dig, X_imp) = readDFW.getAllTrainData(FLAGS.dataDirPrefix, FLAGS.trainImagesDir, IMAGERES, conversionModel)
	(X_plain_raw, X_dig_raw) = readDFW.getRawTrainData(FLAGS.dataDirPrefix, FLAGS.trainImagesDir, IMAGERES)

	# Split X_dig person-wise for pretraining & framework data
	(X_dig_pre, _) = readDFW.splitDisguiseData(X_dig, pre_ratio=0.5)
	(_, X_dig_post) = readDFW.splitDisguiseData(X_dig_raw, pre_ratio=0.5)

	ensemble = [siamese.SiameseNetwork(FEATURERES, "models/ensemble1", 0.1)]
	ensembleNoise = [noise.Gaussian() for _ in ensemble]
	#ensembleNoise = [noise.Noise() for _ in ensemble]

	# Ready committee of models
	bag = committee.Bagging(ensemble, ensembleNoise)
	disguisedFacesModel = siamese.SiameseNetwork(FEATURERES, "models/disguisedModel", 0.1)
	
	# Create generators
	normGen = readDFW.getNormalGenerator(X_plain, FLAGS.batch_size)
	disgGen = readDFW.getNormalGenerator(X_dig_pre, FLAGS.batch_size)
	normImpGen = readDFW.getNormalGenerator(X_imp, FLAGS.batch_size)
	impGenNorm  = readDFW.getImposterGenerator(X_plain, X_imp, FLAGS.batch_size)
	impGenDisg = readDFW.getImposterGenerator(X_dig_pre, X_imp, FLAGS.batch_size)

	# Train/Finetune disguised-faces model
	if FLAGS.refine_models:
		disguisedFacesModel.maybeLoadFromMemory()
		dataGen = readDFW.getGenerator(disgGen, normImpGen, impGenDisg, FLAGS.batch_size, 0)
                disguisedFacesModel.customTrainModel(dataGen, FLAGS.dig_epochs, FLAGS.batch_size, 0.2)
                disguisedFacesModel.save()
		print('Finetuned disguised-faces model')
	elif not disguisedFacesModel.maybeLoadFromMemory():
		dataGen = readDFW.getGenerator(disgGen, normImpGen, impGenDisg, FLAGS.batch_size, 0)
		disguisedFacesModel.customTrainModel(dataGen, FLAGS.dig_epochs, FLAGS.batch_size, 0.2)
		disguisedFacesModel.save()
		print('Trained disguised-faces model')
	else:
		print('Loaded disguised-faces model from memory')

	# Train/Finetune undisguised model(s), if not already trained
	for individualModel in ensemble:
		if FLAGS.refine_models:
			individualModel.maybeLoadFromMemory()
			dataGen = readDFW.getGenerator(normGen, normImpGen, impGenNorm, FLAGS.batch_size, 0)
                        individualModel.customTrainModel(dataGen, FLAGS.undig_epochs, FLAGS.batch_size, 0.2)
                        individualModel.save()
		elif not individualModel.maybeLoadFromMemory():
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

	# Save  models for different values of active ratio
	active_values = [0.1, 0.25, 0.5, 0.75, 1]
	active_index = 0

	for ii in range(0, len(X_dig_post), FRAMEWORK_BS):
		print( (ii / FRAMEWORK_BS) + 1, "iteration")
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
		noisy_data = [ bag.attackModel(p, IMAGERES) for p in batch_x]

		# Get features back from noisy images
                noisy_data = [ conversionModel.process(p) for p in noisy_data ]

		# Pass these to disguised-faces model, get predictions
		disguisedPredictions = disguisedFacesModel.predict(noisy_data)

		# Pick examples that were misclassified
		misclassifiedIndices = []
		for j in range(len(disguisedPredictions)):
			c1 = disguisedPredictions[j][0] >= 0.5
			c2 = ensemblePredictions[j][0] >= 0.5
			#print disguisedPredictions[j][0], ensemblePredictions[j][0]
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
			if FLAGS.augment:
				batch_x_aug, batch_y_aug = helpers.augment_data([batch_x[0][queryIndices], batch_x[1][queryIndices]], helpers.roundoff(intermediate), 1)
				batch_x_aug_features = [conversionModel.process(p) for p in batch_x_aug]
				train_df_left_x = np.concatenate((train_df_left_x, batch_x_aug_features[0], X_old_left))
				train_df_right_x = np.concatenate((train_df_right_x, batch_x_aug_features[1], X_old_right))
				train_df_y = np.concatenate((train_df_y, batch_y_aug, Y_old))
			else:
				train_df_left_x = np.concatenate((train_df_left_x, batch_x_features[0][queryIndices], X_old_left))
				train_df_right_x = np.concatenate((train_df_right_x, batch_x_features[1][queryIndices], X_old_right))
				train_df_y = np.concatenate((train_df_y, helpers.roundoff(intermediate), Y_old))
			# Use a lower learning rate for finetuning ?
			disguisedFacesModel.finetune([train_df_left_x, train_df_right_x], train_df_y, FLAGS.ft_epochs, 16, 1)
			train_df_left_x = np.array([])
			train_df_right_x = np.array([])
			train_df_y = np.array([])

		print("Check if %d less than %d" % (int(active_values[active_index] * 31372), ACTIVE_COUNT))
		# Stop algorithm if limit reached/exceeded
		if int(active_values[active_index] * 31372) <= ACTIVE_COUNT:
			disguisedFacesModel.save(FLAGS.out_model + str(100 * active_values[active_index]))
			print("Saved for ratio %f" % active_values[active_ratio])
			active_index += 1
			if active_index >= len(active_values):
				break
