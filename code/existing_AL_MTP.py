import readDFW3 as readDFW
import readMTP3 as readMTP
import siamese3 as siamese

from learners import ActiveLearner
import uncertainty

from keras_wrapper import KerasClassifier
from keras.models import load_model

import numpy as np
import tensorflow as tf
import keras
from tensorflow.python.platform import flags

# Set seed for reproducability
tf.set_random_seed(42)

# Don't hog GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
keras.backend.set_session(sess)

# Global
IMAGERES = (48, 48)
FEATURERES = (2048,)

FLAGS = flags.FLAGS
flags.DEFINE_string('dataDirPrefix', 'DFW/DFW_Data/', 'Path to DFW data directory')
flags.DEFINE_string('trainImagesDir', 'Training_data', 'Path to DFW training-data images')

flags.DEFINE_string('model_path', 'WACV_models/active', 'Prefix for model')
flags.DEFINE_string('out_model', 'WACV_models/post_active', 'Name of model to be saved after active-learning')
flags.DEFINE_string('query_strategy', 'uncertainty_sampling', 'Query strategy for active-learning')

flags.DEFINE_integer('lowRes', 48, 'Resolution for low-res model (X,X)')
flags.DEFINE_integer('epochs', 3, 'Number of epochs while training model')
flags.DEFINE_integer('batch_size', 512, 'Batch size while training model')

flags.DEFINE_float('split_ratio', 0.3, 'How much of data to use for training M2')
flags.DEFINE_float('active_ratio', 1.0, 'Upper cap on ratio of unlabelled examples to be querried for labels')


def get_strategy_object():
	if FLAGS.query_strategy == 'uncertainty_sampling':
		return uncertainty.uncertainty_sampling
	elif FLAGS.query_strategy == 'margin_sampling':
		return uncertainty.margin_sampling
	elif FLAGS.query_strategy == 'entropy_sampling':
		return uncertainty.entropy_sampling


if __name__ == "__main__":
	# Define image featurization model

	IMAGERES = (FLAGS.lowRes, FLAGS.lowRes)
	
	# Construct ensemble of models
	model = siamese.SiameseNetwork(FEATURERES, FLAGS.model_path, 0.1)

	# Load images, convert to feature vectors for faster processing
	(X_plain, _, X_imp) = readDFW.getAllTrainData(FLAGS.dataDirPrefix,
													FLAGS.trainImagesDir,
													IMAGERES,
													conversionModel,
													combine_normal_imp=True)

	(X_plain_pre, X_plain_post) = readDFW.splitDisguiseData(X_plain, pre_ratio=FLAGS.split_ratio)

	# Pre-train model if not already trained
	if not model.maybeLoadFromMemory():
		normGen     = readDFW.getNormalGenerator(X_plain_pre, FLAGS.batch_size)
		normImpGen  = readDFW.getNormalGenerator(X_imp, FLAGS.batch_size)
		impGenNorm  = readDFW.getImposterGenerator(X_plain_pre, X_imp, FLAGS.batch_size)

		dataGen = readDFW.getGenerator(normGen, normImpGen, impGenNorm, FLAGS.batch_size, 0)
		model.customTrainModel(dataGen, FLAGS.epochs, FLAGS.batch_size, 0.2)
		model.save()
		exit()

	# Create generators
	normGen     = readDFW.getNormalGenerator(X_plain_post, FLAGS.batch_size, infinite=False)
	normImpGen  = readDFW.getNormalGenerator(X_imp, FLAGS.batch_size, infinite=False)
	impGenNorm  = readDFW.getImposterGenerator(X_plain_post, X_imp, FLAGS.batch_size, infinite=False)
	dataGen = readDFW.getGenerator(normGen, normImpGen, impGenNorm, 4 * FLAGS.batch_size, 0)

	def dummy_fn():
		return model.siamese_net

	# SKLearn wrapper around model
	wrapped_model = KerasClassifier(dummy_fn)

	# initialize ActiveLearner
	learner = ActiveLearner(
		estimator=wrapped_model,
		query_strategy=get_strategy_object(),
		verbose=1)

	n_queries = 0
	while True:
		(X_old_left, X_old_right), Y_old = next(dataGen)
		if X_old_left is None:
			break
		query_idx, query_instance = learner.query([X_old_left, X_old_right],
													n_instances=int(len(X_old_left) * FLAGS.active_ratio),
													verbose=0)
		learner.teach(
			X=[X_old_left[query_idx], X_old_right[query_idx]],
			y=Y_old[query_idx],
			only_new=True,
			verbose=1, epochs=2,
			validation_split=0.1,
		)
		n_queries += 1

	# Save model
	model.siamese_net.save_weights(FLAGS.out_model + ".h5")
