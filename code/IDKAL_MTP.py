import readMTP, readDFW
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


def init():
	# Set seed for reproducability
	tf.set_random_seed(42)

	# Don't hog GPU
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)
	keras.backend.set_session(sess)

	# Set low image resolution
	assert(FLAGS.lowRes <= GlobalConstants.normal_res)
	GlobalConstants.low_res = (FLAGS.lowRes, FLAGS.lowRes)


class GlobalConstants:
	image_res = (224, 224)
	feature_res = (2048,)
	normal_res = (150, 150)
	low_res = (32,32)
	# feature_res = (25088,)
	active_count = 0

FLAGS = flags.FLAGS

flags.DEFINE_string('dataDirPrefix', '../../MultiPie51/train', 'Path to MTP data directory')
flags.DEFINE_string('out_model', 'MTP_models/postIDKAL', 'Name of model to be saved after finetuning')
flags.DEFINE_string('ensemble_basepath', 'MTP_models/ensemble', 'Prefix for ensemble models')
flags.DEFINE_string('lowres_basemodel', 'MTP_models/lowresModel', 'Name for model trained on low-res faces')
flags.DEFINE_string('noise', 'gaussian,saltpepper,poisson', 'Prefix for ensemble models')

flags.DEFINE_integer('lowRes', 32, 'Resolution for low-res model (X,X)')
flags.DEFINE_integer('ft_epochs', 3, 'Number of epochs while finetuning model')
flags.DEFINE_integer('batch_size', 16, 'Batch size while sampling from unlabelled data')
flags.DEFINE_integer('lowres_epochs', 4, 'Number of epochs while training lowres-faces model')
flags.DEFINE_integer('highres_epochs', 4, 'Number of epochs while fine-tuning highres-faces model')
flags.DEFINE_integer('batch_send', 64, 'Batch size while finetuning disguised-faces model')
flags.DEFINE_integer('mixture_ratio', 1, 'Ratio of unperturbed:perturbed examples while finetuning network')
flags.DEFINE_integer('idkal_bs', 16, 'Batch size to be used while running framework')
flags.DEFINE_integer('num_ensemble_models', 1, 'Number of models to use in ensemble for highres-faces')

flags.DEFINE_float('active_ratio', 1.0, 'Upper cap on ratio of unlabelled examples to be querried for labels')
flags.DEFINE_float('split_ratio', 0.5, 'How much of disguised-face data to use for training M2')
flags.DEFINE_float('disparity_ratio', 0.25, 'What percentage of data to pick to pass on to oracle')
flags.DEFINE_float('eps', 0.1, 'Region around equiboundary for even considering querying the oracle')

flags.DEFINE_boolean('augment', False, 'Augment data while finetuning covariate-based model?')
flags.DEFINE_boolean('refine_models', False, 'Refine previously trained models?')
flags.DEFINE_boolean('train_lowres_model', False, 'Train lowres-face model? (quits after training)')
flags.DEFINE_boolean('blind_strategy', False, 'If yes, pick all where dispary >= 0.5, otherwise pick according to disparity_ratio')


if __name__ == "__main__":
	# Reproducability
	init()

	# Set resolution according to flag
	GlobalConstants.low_res = (FLAGS.lowRes, FLAGS.lowRes)
	print("Low resolution : %s" % str(GlobalConstants.low_res))

	# Define image featurization model
	conversionModel = siamese.RESNET50(GlobalConstants.image_res)

	# Load images, convert to feature vectors for faster processing
	X_dig_raw = readMTP.readAllImages(FLAGS.dataDirPrefix)

	# Some sanity checks
	assert(0 <= FLAGS.split_ratio and FLAGS.split_ratio <= 1)
	assert(0 <= FLAGS.disparity_ratio and FLAGS.disparity_ratio <= 1)
	assert(0 <= FLAGS.eps and FLAGS.eps < 0.5)
	print("Noise that will be used for IDKAL: %s" % (FLAGS.noise))

	# Set X_dig_post for finetuning second version of model
	if FLAGS.split_ratio > 0:
		(X_dig_pre, X_dig_post) = readDFW.splitDisguiseData(X_dig_raw, pre_ratio=FLAGS.split_ratio)
	elif FLAGS.split_ratio == 1:
		X_dig_pre = X_dig_raw
	else:
		X_dig_post = X_dig_raw

	# Construct ensemble of models
	ensemble = [siamese.SiameseNetwork(GlobalConstants.feature_res, FLAGS.ensemble_basepath + str(i), 0.1) for i in range(1, FLAGS.num_ensemble_models + 1)]
	# Prepare required noises
	desired_noises = FLAGS.noise.split(',')
	ensembleNoise = [noise.get_relevant_noise(x)() for x in desired_noises]

	# Ready committee of models
	bag = committee.Bagging(ensemble, ensembleNoise)
	lowResModel = siamese.SmallRes(GlobalConstants.low_res + (3,), GlobalConstants.feature_res, FLAGS.lowres_basemodel, 0.1)

	if FLAGS.train_lowres_model:
		print('Training lowres-faces model')
		# Create generators for low-res data
		normGen = readDFW.getNormalGenerator(X_dig_pre, FLAGS.batch_size)
		lowResSiamGen = readMTP.getGenerator(normGen, FLAGS.batch_size, 0.2, GlobalConstants.low_res)
		lowResModel.customTrainModel(lowResSiamGen, FLAGS.lowres_epochs, FLAGS.batch_size, 0.2, 32000)
		lowResModel.save()
		exit()
	else:
		lowResModel.maybeLoadFromMemory()
		print('Loaded lowres-faces model from memory')

	normGen = readDFW.getNormalGenerator(X_dig_pre, FLAGS.batch_size)

	# Train/Finetune undisguised model(s), if not already trained
	for individualModel in ensemble:
		if FLAGS.refine_models:
			individualModel.maybeLoadFromMemory()
			dataGen = readMTP.getGenerator(normGen, FLAGS.batch_size, 0.2, GlobalConstants.image_res, conversionModel)
			individualModel.customTrainModel(dataGen, FLAGS.highres_epochs, FLAGS.batch_size, 0.2, 64000)
			individualModel.save()
		elif not individualModel.maybeLoadFromMemory():
			print("Training ensemble model")
			dataGen = readMTP.getGenerator(normGen, FLAGS.batch_size, 0.2, GlobalConstants.image_res, conversionModel)
			individualModel.customTrainModel(dataGen, FLAGS.highres_epochs, FLAGS.batch_size, 0.2, 64000)
			individualModel.save()
	print('Finetuned highres-faces models')

	# Train lowres-faces model only when batch length crosses threshold
	train_lr_left_x  = np.array([])
	train_lr_right_x = np.array([])
	train_lr_y       = np.array([])
	UN_SIZE          = 0

	# Framework begins
	print("== Framework beginning with a pool of %d ==" % (len(pool_X[0])))
	dataGen = readMTP.generatorFeaturized(train_X, train_Y, FLAGS.batch_size, resize_res=GlobalConstants.low_res)
	for ii in range(0, len(pool_X[0]), FLAGS.idkal_bs):
		print("\nIteration #%d" % ((ii / FLAGS.idkal_bs) + 1))

		batch_x_left, batch_x_right, batch_y = pool_X[0][ii: ii + FLAGS.idkal_bs], pool_X[1][ii: ii + FLAGS.idkal_bs], pool_Y[ii: ii + FLAGS.idkal_bs]
		batch_x_left  = np.array(batch_x_left)
		batch_x_right = np.array(batch_x_right)
		batch_x_highres  = readMTP.resizeImages([batch_x_left, batch_x_right], GlobalConstants.image_res)
		batch_x_lowres   = readMTP.resizeImages([batch_x_left, batch_x_right], GlobalConstants.low_res)
		# batch_y here acts as a pseudo-oracle
		# any reference mde to it is counted as a query to the oracle
		UN_SIZE += len(batch_x_left)

		# Get featurized faces to be passed to committee
		batch_x_features = [conversionModel.process(p) for p in batch_x_highres]

		# Get predictions made by committee
		ensemblePredictions = bag.predict(batch_x_features)

		# Get images with added noise
		noisy_data_left  = bag.attackModel(batch_x_left, GlobalConstants.low_res)
		noisy_data_right = bag.attackModel(batch_x_right, GlobalConstants.low_res)
		noisy_data = [noisy_data_left, noisy_data_right]

		# Pass these to disguised-faces model, get predictions
		disguisedPredictions = [lowResModel.predict([noisy_data[0][jj], noisy_data[1][jj]]) for jj in range(len(ensembleNoise))]
		misclassifiedIndices = []
		for dp in disguisedPredictions:
			disparities = []
			for j in range(len(dp)):
				c1 = dp[j][0]
				c2 = ensemblePredictions[j][0]
				# print(c1, c2, "yeehaw")
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
			ensemble_prediction = ensemblePredictions[j][0]
			if ensemble_prediction <= 0.5 - FLAGS.eps or ensemble_prediction >= 0.5 + FLAGS.eps:
				c1 = ensemble_prediction >= 0.5
				c2 = batch_y[j][0] >= 0.5
				print(ensemble_prediction , batch_y[j][0], "hawwyee")
				GlobalConstants.active_count += 1
				if c1 == c2:
					queryIndices.append(j)

		# Log active count so far
		print("== Active Count so far : %d ==" % (GlobalConstants.active_count))

		# If nothing matches, proceed to next set of predictions
		if len(queryIndices) == 0:
			print("== Nothing in this set. Skipping batch ==")
			continue

		intermediate = []
		for i in queryIndices:
			intermediate.append(ensemblePredictions[i][0])
		intermediate = np.array(intermediate)

		mp = int(len(intermediate) / float(len(ensembleNoise)))
		# Gather data to be sent to low res model for training
		if train_lr_y.shape[0] > 0:
			train_lr_left_x  = np.concatenate((train_lr_left_x,)  + tuple([noisy_data[0][i][queryIndices[i*mp :(i+1)*mp]] for i in range(len(ensembleNoise))]))
			train_lr_right_x = np.concatenate((train_lr_right_x,) + tuple([noisy_data[1][i][queryIndices[i*mp :(i+1)*mp]] for i in range(len(ensembleNoise))]))
			train_lr_y       = np.concatenate((train_lr_y,)       + tuple([helpers.roundoff(intermediate)[i*mp:(i+1)*mp]  for i in range(len(ensembleNoise))]))
		else:
			train_lr_left_x  = np.concatenate([noisy_data[0][i][queryIndices[i*mp: (i+1)*mp]] for i in range(len(ensembleNoise))])
			train_lr_right_x = np.concatenate([noisy_data[1][i][queryIndices[i*mp: (i+1)*mp]] for i in range(len(ensembleNoise))])
			train_lr_y       = np.concatenate([helpers.roundoff(intermediate)[i*mp:(i+1)*mp]  for i in range(len(ensembleNoise))])

		print("== Accumulated data so far (for next batch) : %d ==" % train_lr_y.shape[0])
		if train_lr_y.shape[0] >= FLAGS.batch_send:
			# Finetune lowres-faces model with these actively selected data points
			# Also, add unperturbed images to avoid overfitting on noise
			(X_old_left, X_old_right), Y_old = dataGen.next()
			for _ in range(FLAGS.mixture_ratio - 1):
				X_old_temp, Y_old_temp = dataGen.next()
				X_old_left  = np.concatenate((X_old_left,  X_old_temp[0]))
				X_old_right = np.concatenate((X_old_right, X_old_temp[1]))
				Y_old       = np.concatenate((Y_old, Y_old_temp))
			if FLAGS.augment:
				batch_x_aug, batch_y_aug = helpers.augment_data([batch_x_left[queryIndices], batch_x_left[queryIndices]], helpers.roundoff(intermediate), 1)
				train_lr_left_x  = np.concatenate((train_lr_left_x,  batch_x_aug[0], X_old_left))
				train_lr_right_x = np.concatenate((train_lr_right_x, batch_x_aug[1], X_old_right))
				train_lr_y       = np.concatenate((train_lr_y,       batch_y_aug,    Y_old))
			else:
				print(len(train_lr_left_x),  len(batch_x_features[0][queryIndices]), len(X_old_left),  "left")
				print(len(train_lr_right_x), len(batch_x_features[1][queryIndices]), len(X_old_right), "right")
				print(train_lr_left_x.shape, batch_x_features[0][queryIndices].shape, X_old_left.shape, "wut")
				train_lr_left_x  = np.concatenate((train_lr_left_x,  batch_x_features[0][queryIndices], X_old_left))
				train_lr_right_x = np.concatenate((train_lr_right_x, batch_x_features[1][queryIndices], X_old_right))
				train_lr_y       = np.concatenate((train_lr_y,       helpers.roundoff(intermediate),    Y_old))

			# Use a lower learning rate for finetuning ?
			lowResModel.finetune([train_lr_left_x, train_lr_right_x], train_lr_y, FLAGS.ft_epochs, 16, 1)
			train_lr_left_x  = np.array([])
			train_lr_right_x = np.array([])
			train_lr_y       = np.array([])

		# Stop algorithm if limit reached/exceeded
		if int(FLAGS.active_ratio * UN_SIZE) <= GlobalConstants.active_count:
			print("== Specified limit reached! Stopping algorithm ==")
			break

	# Print count of images queried so far
	print("== Active Count: %d out of %d ==" % (GlobalConstants.active_count, UN_SIZE))

	# Save retrained model
	lowResModel.save(FLAGS.out_model)
