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
from sets import Set

# Set seed for reproducability
tf.compat.v1.set_random_seed(42)

# Don't hog GPU
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)
keras.backend.set_session(sess)

# Global
IMAGERES = (224, 224)
# IMAGERES = (112, 112)
FEATURERES = (2048,)
#FEATURERES = (25088,)
#FEATURERES = (512,)
ACTIVE_COUNT = 0

FLAGS = flags.FLAGS

flags.DEFINE_string('dataDirPrefix',       'DFW_Data/',                                       'Path to DFW data directory')
flags.DEFINE_string('trainImagesDir',      'Training_data',                                   'Path to DFW training-data images')
flags.DEFINE_string('testImagesDir',       'Testing_data',                                    'Path to DFW testing-data images')
flags.DEFINE_string('out_model',           'Densenet_models/postALINK',                       'Name of model to be saved after finetuning')
flags.DEFINE_string('ensemble_basepath',   'Densenet_models/ensemble',                        'Prefix for ensemble models')
flags.DEFINE_string('disguised_basemodel', 'Densenet_models/disguisedModel',                  'Name for model trained on disguised faces')
flags.DEFINE_string('noise',               'gaussian,saltpepper,poisson,speckle,adversarial', 'Noise components')

flags.DEFINE_integer('ft_epochs',           3, 'Number of epochs while finetuning model')
flags.DEFINE_integer('batch_size',         16, 'Batch size while sampling from unlabelled data')
flags.DEFINE_integer('dig_epochs',         40, 'Number of epochs while training disguised-faces model')
flags.DEFINE_integer('undig_epochs',       60, 'Number of epochs while fine-tuning undisguised-faces model')
flags.DEFINE_integer('batch_send',         64, 'Batch size while finetuning disguised-faces model')
flags.DEFINE_integer('mixture_ratio',       2, 'Ratio of unperturbed:perturbed examples while finetuning network')
flags.DEFINE_integer('alink_bs',           16, 'Batch size to be used while running framework')
flags.DEFINE_integer('num_ensemble_models', 1, 'Number of models to use in ensemble for undisguised-faces')

flags.DEFINE_float('active_ratio',     1.0, 'Upper cap on ratio of unlabelled examples to be querried for labels')
flags.DEFINE_float('split_ratio',      0.5, 'How much of disguised-face data to use for training M2')
flags.DEFINE_float('disparity_ratio', 0.25, 'What percentage of data to pick to pass on to oracle')
flags.DEFINE_float('eps',             0.05, 'Region around equiboundary for even considering querying the oracle')

flags.DEFINE_boolean('augment',               False, 'Augment data while finetuning covariate-based model?')
flags.DEFINE_boolean('refine_models',         False, 'Refine previously trained models?')
flags.DEFINE_boolean('train_disguised_model', False, 'Train disguised-face model? (quits after training)')
flags.DEFINE_boolean('blind_strategy',        False, 'If yes, pick all where dispary >= 0.5, otherwise pick according to disparity_ratio')


if __name__ == "__main__":
	# Define image featurization model
	conversionModel = siamese.RESNET50(IMAGERES)

	# Load images, convert to feature vectors for faster processing
	(X_plain, X_dig, X_imp)  = readDFW.getAllTrainData(FLAGS.dataDirPrefix, FLAGS.trainImagesDir, IMAGERES, conversionModel)
	(X_plain_raw, X_dig_raw) = readDFW.getRawTrainData(FLAGS.dataDirPrefix, FLAGS.trainImagesDir, IMAGERES)

	# Some sanity checks
	assert(0 <= FLAGS.split_ratio and FLAGS.split_ratio <= 1)
	assert(0 <= FLAGS.disparity_ratio and FLAGS.disparity_ratio <= 1)
	assert(0 <= FLAGS.eps and FLAGS.eps < 0.5)
	print(">> Noise that will be used for ALINK: %s" % (FLAGS.noise))

	# Set X_dig_post for finetuning second version of model
	if FLAGS.split_ratio > 0:
		(X_dig_pre, _)  = readDFW.splitDisguiseData(X_dig, pre_ratio=FLAGS.split_ratio)
		(_, X_dig_post) = readDFW.splitDisguiseData(X_dig_raw, pre_ratio=FLAGS.split_ratio)
	elif FLAGS.split_ratio == 1:
		X_dig_pre = X_dig_raw
	else:
		X_dig_post = X_dig_raw

	# Ready disguised face model
	disguisedFacesModel = siamese.SiameseNetwork(FEATURERES, FLAGS.disguised_basemodel, 0.1)

	# Prepare required noises
	desired_noises = FLAGS.noise.split(',')
	ensembleNoise  = [noise.get_relevant_noise(x)(model=disguisedFacesModel, sess=sess, feature_model=conversionModel) for x in desired_noises]

	# Construct ensemble of models
	ensemble = [siamese.SiameseNetwork(FEATURERES, FLAGS.ensemble_basepath + str(i), 0.1) for i in range(1, FLAGS.num_ensemble_models+1)]
	bag      = committee.Bagging(ensemble, ensembleNoise)

	if FLAGS.train_disguised_model:
		# Create generators for disguised model
		normGen     = readDFW.getNormalGenerator(X_dig_pre, FLAGS.batch_size)
		normImpGen  = readDFW.getNormalGenerator(X_imp, FLAGS.batch_size)
		impGenNorm  = readDFW.getImposterGenerator(X_dig_pre, X_imp, FLAGS.batch_size)

		# Train/Finetune disguised-faces model
		print('>> Training disguised-faces model')
		dataGen   = readDFW.getGenerator(normGen, normImpGen, impGenNorm, FLAGS.batch_size, 0)
		disguisedFacesModel.customTrainModel(dataGen, FLAGS.dig_epochs, FLAGS.batch_size, 0.2)
		disguisedFacesModel.save()
		print(">> Trained disguised-faces model. Please restart script without the --train_disguised_model flag")
		exit()
	else:
		disguisedFacesModel.maybeLoadFromMemory()
		print('>> Loaded disguised-faces model from memory')

	# Create generators
	normGen     = readDFW.getNormalGenerator(X_plain, FLAGS.batch_size)
	normImpGen  = readDFW.getNormalGenerator(X_imp, FLAGS.batch_size)
	impGenNorm  = readDFW.getImposterGenerator(X_plain, X_imp, FLAGS.batch_size)

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
	print('>> Finetuned undisguised-faces models')

	# Train disguised-faces model only when batch length crosses threshold
	train_df_left_x  = np.array([])
	train_df_right_x = np.array([])
	train_df_y       = np.array([])

	# Do something about catastrophic forgetting (?)
	UN_SIZE = 0

	# Framework begins
	print(">> Framework beginning with a pool of %d" % (len(X_dig_post)))
	dataGen = readDFW.getGenerator(normGen, normImpGen, impGenNorm, FLAGS.batch_size, 0)
	for ii in range(0, len(X_dig_post), FLAGS.alink_bs):
		print("\nIteration #%d" % ((ii / FLAGS.alink_bs) + 1))
		plain_part    = X_plain_raw[ii: ii + FLAGS.alink_bs]
		disguise_part = X_dig_post[ii: ii + FLAGS.alink_bs]

		# Create pairs of images
		batch_x, batch_y = readDFW.createMiniBatch(plain_part, disguise_part)
		# batch_y here acts as a pseudo-oracle
		# any reference made to it is counted as a query to the oracle
		UN_SIZE += len(batch_x[0])

		# Get featurized faces to be passed to committee
		batch_x_features = [conversionModel.process(p) for p in batch_x]

		# Get predictions made by committee
		ensemblePredictions = bag.predict(batch_x_features)

		# Get images with added noise
		m1_labels  = keras.utils.to_categorical(np.argmax(ensemblePredictions, axis=1), 2)
		noisy_data = bag.attackModel(batch_x, IMAGERES, m1_labels)

		# Get features back from noisy images
		noisy_data = [[conversionModel.process(p) for p in part] for part in noisy_data]

		# Pass these to disguised-faces model, get predictions
		disguisedPredictions = [disguisedFacesModel.predict([noisy_data[0][jj], noisy_data[1][jj]]) for jj in range(len(ensembleNoise))]
		misclassifiedIndices = []
		for dp in disguisedPredictions:
			disparities = []
			for j in range(len(dp)):
				c1 = dp[j][1]
				c2 = ensemblePredictions[j][1]
				if FLAGS.blind_strategy:
					if (c1 >= 0.5) != (c2 >= 0.5):
						disparities.append(j)
				else:
					disparities.append(-np.absolute(c1 - c2))
			if not FLAGS.blind_strategy:
				disparities = np.argsort(disparities)[:int(len(disparities) * FLAGS.disparity_ratio)]
			misclassifiedIndices.append(disparities)
		# Pick cases where all noise makes the model flip predictions (according to criteria)
		all_noise_works = Set(misclassifiedIndices[0])
		for j in range(1, len(misclassifiedIndices)):
			all_noise_works = all_noise_works & Set(misclassifiedIndices[j])
		misclassifiedIndices = list(all_noise_works)

		# Query oracle, pick examples for which ensemble was (crudely) right
		queryIndices = []
		for j in misclassifiedIndices:
			# If ensemble's predictions not in grey area:
			ensemble_prediction = ensemblePredictions[j][1]
			if ensemble_prediction <= 0.5 - FLAGS.eps or ensemble_prediction >= 0.5 + FLAGS.eps:
				c1 = ensemble_prediction >= 0.5
				c2 = batch_y[j][0] >= 0.5
				ACTIVE_COUNT += 1
				if c1 == c2:
					queryIndices.append(j)

		# Log active count so far
		print("Active Count so far : %d" % (ACTIVE_COUNT))

		# If nothing matches, proceed to next set of predictions
		if len(queryIndices) == 0:
			continue

		intermediate = []
		for i in queryIndices:
			intermediate.append(ensemblePredictions[i][1])
		intermediate = np.array(intermediate)

		# Create equal partitions while mixing multiple types of noise together
		mp = int(len(intermediate) / float(len(ensembleNoise)))
		# Gather data to be sent to low res model for training
		if train_df_y.shape[0] > 0:
			train_df_left_x  = np.concatenate((train_df_left_x,)  + tuple([noisy_data[0][i][queryIndices[i*mp:(i+1)*mp]] for i in range(len(ensembleNoise))]))
			train_df_right_x = np.concatenate((train_df_right_x,) + tuple([noisy_data[1][i][queryIndices[i*mp:(i+1)*mp]] for i in range(len(ensembleNoise))]))
			train_df_y       = np.concatenate((train_df_y,)       + tuple([helpers.roundoff(intermediate)[i*mp:(i+1)*mp] for i in range(len(ensembleNoise))]))
		else:
			train_df_left_x  = np.concatenate([noisy_data[0][i][queryIndices[i*mp:(i+1)*mp]] for i in range(len(ensembleNoise))])
			train_df_right_x = np.concatenate([noisy_data[1][i][queryIndices[i*mp:(i+1)*mp]] for i in range(len(ensembleNoise))])
			train_df_y       = np.concatenate([helpers.roundoff(intermediate)[i*mp:(i+1)*mp] for i in range(len(ensembleNoise))])

		if train_df_y.shape[0] >= FLAGS.batch_send:
			# Finetune disguised-faces model with this actively selected data points
			# Also add unperturbed images to avoid overfitting on noise
			(X_old_left, X_old_right), Y_old = dataGen.next()
			for _ in range(FLAGS.mixture_ratio - 1):
				X_old_temp, Y_old_temp = dataGen.next()
				X_old_left  = np.concatenate((X_old_left,  X_old_temp[0]))
				X_old_right = np.concatenate((X_old_right, X_old_temp[1]))
				Y_old       = np.concatenate((Y_old, Y_old_temp))
			if FLAGS.augment:
				batch_x_aug, batch_y_aug = helpers.augment_data([batch_x[0][queryIndices], batch_x[1][queryIndices]], helpers.roundoff(intermediate), 1)
				batch_x_aug_features     = [conversionModel.process(p) for p in batch_x_aug]
				train_df_left_x          = np.concatenate((train_df_left_x,  batch_x_aug_features[0], X_old_left))
				train_df_right_x         = np.concatenate((train_df_right_x, batch_x_aug_features[1], X_old_right))
				train_df_y               = np.concatenate((train_df_y, batch_y_aug, Y_old))
			else:
				# print(len(train_df_left_x), len(batch_x_features[0][queryIndices]), len(X_old_left), "left")
				# print(len(train_df_right_x), len(batch_x_features[1][queryIndices]), len(X_old_right), "right")
				train_df_left_x  = np.concatenate((train_df_left_x,  batch_x_features[0][queryIndices], X_old_left))
				train_df_right_x = np.concatenate((train_df_right_x, batch_x_features[1][queryIndices], X_old_right))
				train_df_y       = np.concatenate((train_df_y, helpers.roundoff(intermediate), Y_old))

			# Use a lower learning rate for finetuning ?
			disguisedFacesModel.finetune([train_df_left_x, train_df_right_x], train_df_y, FLAGS.ft_epochs, 16, 1)
			train_df_left_x = np.array([])
			train_df_right_x = np.array([])
			train_df_y = np.array([])

		# Stop algorithm if limit reached/exceeded
		# if int((FLAGS.active_ratio * len(X_dig_post) * (FLAGS.alink_bs - 1)) / 2) <= ACTIVE_COUNT:
		if int(FLAGS.active_ratio * UN_SIZE) <= ACTIVE_COUNT:
			print(">> Specified limit reached! Stopping algorithm")
			break

	# Print count of images queried so far
	print(">> Active Count: %d out of %d" % (ACTIVE_COUNT, UN_SIZE))

	# Save retrained model
	disguisedFacesModel.save(FLAGS.out_model)
