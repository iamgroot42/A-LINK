import load_data
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
DATARES = (150, 150)
HIGHRES = (224, 224)
LOWRES = (32, 32)
FEATURERES = (2048,)
FRAMEWORK_BS = 16
ACTIVE_COUNT = 0

FLAGS = flags.FLAGS

flags.DEFINE_string('imagesDir', 'data/', 'Path to all images')
flags.DEFINE_string('lowResImagesDir', 'data_final/lowres/', 'Path to low-res images')
flags.DEFINE_string('highResImagesDir', 'data_final/highres/', 'Path to high-res images')
flags.DEFINE_string('unlabelledList', 'fileLists/unlabelledData.txt', 'Path to unlabelled images list')
flags.DEFINE_string('testDataList', 'fileLists/testData.txt', 'Path to test images list')
flags.DEFINE_integer('batch_size', 16, 'Batch size while sampling from unlabelled data')
flags.DEFINE_integer('low_epochs', 50, 'Number of epochs while training low-resolution model')
flags.DEFINE_integer('high_epochs', 20, 'Number of epochs while fine-tuning high-resolution model')
flags.DEFINE_integer('batch_send', 64, 'Batch size while finetuning disguised-faces model')
flags.DEFINE_float('active_ratio', 1.0, 'Upper cap on ratio of unlabelled examples to be qurried for labels')
flags.DEFINE_string('out_model', 'models/fineTuned', 'Name of model to be saved after finetuning')


if __name__ == "__main__":
	# Define image featurization model
	conversionModel = siamese.RESNET50(HIGHRES)

	# Initialize generator for unlabelled data
	unlabelledImagesGenerator = load_data.getUnlabelledData(FLAGS.imagesDir, FLAGS.unlabelledList, FLAGS.batch_size)

	# Initialize generator for high-resolution data
	highgenTrain, highgenVal = load_data.returnGenerators(FLAGS.highResImagesDir + "train", FLAGS.highResImagesDir + "val", HIGHRES, 16, helpers.hr_preprocess)

	# Get high-gen siamese generator from current generators
	highgenSiam = load_data.combineGenSiam(highgenTrain, highgenVal, conversionModel, FLAGS.batch_size)

	# Load low-resolution data
	(X_low, Y_low) = load_data.resizeLoadDataAll(FLAGS.imagesDir, FLAGS.lowResImagesDir + "train", FLAGS.lowResImagesDir + "val", LOWRES)
	lowResSiamGen = load_data.combineGenSiam(load_data.dataToSiamGen(X_low, Y_low, 16), None, None, 16)

	ensemble = [siamese.SiameseNetwork(FEATURERES, "ensemble1_multipie", 0.1)]
	#ensembleNoise = [noise.Gaussian() for _ in ensemble]
	ensembleNoise = [noise.Noise() for _ in ensemble]

	# Ready committee of models
        bag = committee.Bagging(ensemble, ensembleNoise)
	lowResModel = siamese.SmallRes(LOWRES + (3,), FEATURERES, "lowResModel_multipie", 0.1)

	# Train low-resolution model
	if not lowResModel.maybeLoadFromMemory():
		# Get siamese generator from low-res data
		lowResModel.customTrainModel(lowResSiamGen, FLAGS.low_epochs, FLAGS.batch_size, 0.2)
		print('Trained low resolution model')
		lowResModel.save()
	else:
		print("Loaded", lowResModel.modelName, "from memory")

	# Finetune high-resolution model(s), if not already trained
	for individualModel in ensemble:
		if not individualModel.maybeLoadFromMemory():
			individualModel.customTrainModel(highgenSiam, FLAGS.high_epochs, FLAGS.batch_size, 0.2)
			individualModel.save()
		else:
			print("Loaded", individualModel.modelName, "from memory")

	# Train low res model only when batch length crosses threshold
	train_lr_left_x = np.array([])
	train_lr_right_x = np.array([])
	train_lr_y = np.array([])
	UN_SIZE = load_data.getContentsSize(FLAGS.testDataList)

	# Framework begins
	while True:
		try:
			batch_x, batch_y = unlabelledImagesGenerator.next()
		except:
			break
		
		batch_x_hr = load_data.resize(batch_x, HIGHRES)
		batch_x_lr = load_data.resize(batch_x, LOWRES)

		# Create pairs of images
		batch_x_hr, _ = load_data.labelToSiamese(batch_x_hr, batch_y)
		batch_x_lr, batch_y = load_data.labelToSiamese(batch_x_lr, batch_y)
		UN_SIZE += len(batch_x_hr[0])
		
		# Get featurized faces to be passed to committee
		batch_x_features = [ conversionModel.process(p) for p in batch_x_hr]

		# Get predictions made by committee
		ensemblePredictions = bag.predict(batch_x_features)

		# Get images with added noise
		noisy_data = [ bag.attackModel(p, LOWRES) for p in batch_x_hr]

		# Pass these to disguised-faces model, get predictions
		lowResPredictions = lowResModel.predict(noisy_data)

		# Pick examples that were misclassified
		misclassifiedIndices = []
		for j in range(len(lowResPredictions)):
			c1 = lowResPredictions[j][0] >= 0.5
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
		if train_lr_y.shape[0] > 0:
			train_lr_left_x = np.concatenate((train_lr_left_x, noisy_data[0][queryIndices]))
			train_lr_right_x = np.concatenate((train_lr_right_x, noisy_data[1][queryIndices]))
			train_lr_y = np.concatenate((train_lr_y, helpers.roundoff(intermediate)))
		else:
			train_lr_left_x = np.copy(noisy_data[0][queryIndices])
			train_lr_right_x = np.copy(noisy_data[1][queryIndices])
			train_lr_y = np.copy(helpers.roundoff(intermediate))

		if train_lr_y.shape[0] >= FLAGS.batch_send:
			# Finetune low-resolution model with this actively selected data points
			# Also add unperturbed images to avoid overfitting on noise
			train_lr_left_x = np.concatenate((train_lr_left_x, batch_x_lr[0][queryIndices]))
			train_lr_right_x = np.concatenate((train_lr_right_x, batch_x_lr[1][queryIndices]))
			train_lr_y = np.concatenate((train_lr_y, helpers.roundoff(intermediate)))
			# Use a lower learning rate for finetuning ?
			lowResModel.finetune([train_lr_left_x, train_lr_right_x], train_lr_y, 3, 16, 1)
			train_df_left_x = np.array([])
			train_df_right_x = np.array([])
			train_df_y = np.array([])

		print(FLAGS.active_ratio, UN_SIZE, (FRAMEWORK_BS-1), ACTIVE_COUNT)

		# Stop algorithm if limit reached/exceeded
		if int((FLAGS.active_ratio * UN_SIZE * (FRAMEWORK_BS - 1)) / 2) <= ACTIVE_COUNT:
			break

	# Print count of images queried so far
	print("Active Count:", ACTIVE_COUNT, "out of:", UN_SIZE)

	# Save retrained model
	lowResModel.save(FLAGS.out_model)
