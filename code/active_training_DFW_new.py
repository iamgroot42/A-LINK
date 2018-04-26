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
FRAMEWORK_BS = 8
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
flags.DEFINE_float('active_ratio', 1.0, 'Upper cap on ratio of unlabelled examples to be qurried for labels')
flags.DEFINE_integer('mixture_ratio', 1, 'Ratio of unperturbed:perturbed examples while finetuning network')
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

	# Set X_dig_post for finetuning second version of model
	X_dig_post = X_dig_raw

	ensemble = [siamese.SiameseNetwork(FEATURERES, "models/ensemble1", 0.1)]
	#ensembleNoise = [noise.Gaussian() for _ in ensemble]
	#ensembleNoise = [noise.Noise() for _ in ensemble]
	#ensembleNoise = [noise.SaltPepper() for _ in ensemble]
	#ensembleNoise = [noise.Poisson() for _ in ensemble]
	#ensembleNoise = [noise.Speckle() for _ in ensemble]
	ensembleNoise = [noise.Gaussian(), noise.SaltPepper(), noise.Speckle()]

	# Ready committee of models
	bag = committee.Bagging(ensemble, ensembleNoise)
	disguisedFacesModel = siamese.SiameseNetwork(FEATURERES, "models/ensemble1_backup", 0.1)
	
	# Create generators
	normGen = readDFW.getNormalGenerator(X_plain, FLAGS.batch_size)
	normImpGen = readDFW.getNormalGenerator(X_imp, FLAGS.batch_size)
	impGenNorm  = readDFW.getImposterGenerator(X_plain, X_imp, FLAGS.batch_size)

	# Train/Finetune disguised-faces model
	disguisedFacesModel.maybeLoadFromMemory()
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
                #noisy_data = [ conversionModel.process(p) for p in noisy_data ]

		# Pass these to disguised-faces model, get predictions
		#disguisedPredictions = disguisedFacesModel.predict(noisy_data)

		noisy_data = [ [conversionModel.process(p) for p in part] for part in noisy_data]
		disguisedPredictions = [disguisedFacesModel.predict([noisy_data[0][jj], noisy_data[1][jj]]) for jj in range(3)] 
		misclassifiedIndices = []
		for dp in disguisedPredictions:
			tortoise = []
			for j in range(len(dp)):
				c1 = dp[j][0] #>= 0.5
				c2 = ensemblePredictions[j][0] #>= 0.5
				#if c1 != c2:
				#	tortoise.append(j)
				tortoise.append(-np.absolute(c1 - c2))
			tortoise = np.argsort(tortoise)[:len(tortoise) / 8]
			misclassifiedIndices.append(tortoise)
			#print(tortoise)
		turtle = Set(misclassifiedIndices[0])
		for j in range(1, len(misclassifiedIndices)):
			turtle = turtle & Set(misclassifiedIndices[j])
		misclassifiedIndices = list(turtle)
		print(misclassifiedIndices)

		# Pick examples that were misclassified (sort according to magnitude of change in score, pick top eighth)
		#misclassifiedIndices = []
		#for j in range(len(disguisedPredictions)):
		#	c1 = disguisedPredictions[j][0]
		#	c2 = ensemblePredictions[j][0]
		#	misclassifiedIndices.append(-np.absolute(c1 - c2))
		#misclassifiedIndices = np.argsort(misclassifiedIndices)[:len(misclassifiedIndices) / 4]

		# Query oracle, pick examples for which ensemble was (crudely) right
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

		mp = int(0.3 * len(intermediate))
		# Gather data to be sent to low res model for training
		if train_df_y.shape[0] > 0:
			#train_df_left_x = np.concatenate((train_df_left_x, noisy_data[0][queryIndices]))
			#train_df_left_x = np.concatenate((train_df_left_x, batch_x_features[0][queryIndices]))
			train_df_left_x = np.concatenate((train_df_left_x, noisy_data[0][0][queryIndices[:mp]], noisy_data[0][1][queryIndices[mp:2*mp]], noisy_data[0][2][queryIndices[2*mp:]]))
			#train_df_right_x = np.concatenate((train_df_right_x, noisy_data[1][queryIndices]))
			#train_df_right_x = np.concatenate((train_df_right_x, batch_x_features[1][queryIndices]))
			train_df_right_x = np.concatenate((train_df_right_x, noisy_data[1][0][queryIndices[:mp]], noisy_data[1][1][queryIndices[mp:2*mp]], noisy_data[1][2][queryIndices[2*mp:]]))
			#train_df_y = np.concatenate((train_df_y, helpers.roundoff(intermediate)))
			#print len(queryIndices[:mp]), len(queryIndices[mp:2*mp]), len(queryIndices[2*mp:]), "X after"
			train_df_y = np.concatenate((train_df_y, helpers.roundoff(intermediate)[:mp], helpers.roundoff(intermediate)[mp:2*mp], helpers.roundoff(intermediate)[2*mp:]))
			#print len(helpers.roundoff(intermediate)[:mp]), len(helpers.roundoff(intermediate)[mp:2*mp]), len(helpers.roundoff(intermediate)[2*mp:]), "Y after"
		else:
			#train_df_left_x = np.copy(noisy_data[0][queryIndices])
			#train_df_left_x = np.copy(batch_x_features[0][queryIndices])
			train_df_left_x = np.concatenate((noisy_data[0][0][queryIndices[:mp]], noisy_data[0][1][queryIndices[mp:2*mp]], noisy_data[0][2][queryIndices[2*mp:]]))
			#train_df_right_x = np.copy(noisy_data[1][queryIndices])
			train_df_right_x = np.concatenate((noisy_data[1][0][queryIndices[:mp]], noisy_data[1][1][queryIndices[mp:2*mp]], noisy_data[1][2][queryIndices[2*mp:]]))
			#train_df_y = np.copy(helpers.roundoff(intermediate))
			train_df_y = np.concatenate((helpers.roundoff(intermediate)[:mp], helpers.roundoff(intermediate)[mp:2*mp], helpers.roundoff(intermediate)[2*mp:]))
			#print len(queryIndices[:mp]), len(queryIndices[mp:2*mp]), len(queryIndices[2*mp:]), "X before"
			#print len(helpers.roundoff(intermediate)[:mp]), len(helpers.roundoff(intermediate)[mp:2*mp]), len(helpers.roundoff(intermediate)[2*mp:]), "Y before"

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

		# Stop algorithm if limit reached/exceeded
		#if int((FLAGS.active_ratio * len(X_dig_post) * (FRAMEWORK_BS - 1)) / 2) <= ACTIVE_COUNT:
		#	break

	# Print count of images queried so far
	print("Active Count:", ACTIVE_COUNT, "out of:", UN_SIZE)

	# Save retrained model
	disguisedFacesModel.save(FLAGS.out_model)
