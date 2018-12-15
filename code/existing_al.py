import readDFW3 as readDFW
import siamese3 as siamese

# from modAL.models import ActiveLearner
from learners import ActiveLearner
# from modAL.uncertainty import uncertainty_sampling
from uncertainty import uncertainty_sampling

# from keras.wrappers.scikit_learn import KerasClassifier
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
IMAGERES = (224, 224)
FEATURERES = (2048,)

FLAGS = flags.FLAGS

flags.DEFINE_string('dataDirPrefix', 'DFW/DFW_Data/', 'Path to DFW data directory')
flags.DEFINE_string('trainImagesDir', 'Training_data', 'Path to DFW training-data images')
flags.DEFINE_string('model_path', 'WACV_models/active', 'Prefix for model')
flags.DEFINE_string('out_model', 'WACV_models/post_active', 'Name of model to be saved after active-learning')

flags.DEFINE_integer('epochs', 3, 'Number of epochs while training model')
flags.DEFINE_integer('batch_size', 16, 'Batch size while training model')
flags.DEFINE_integer('query_batchsize', 16, 'Batch size to be used while running IDKAL')

flags.DEFINE_float('split_ratio', 0.5, 'How much of data to use for training M2')
flags.DEFINE_float('active_ratio', 1.0, 'Upper cap on ratio of unlabelled examples to be querried for labels')


if __name__ == "__main__":
	# Define image featurization model
	conversionModel = siamese.RESNET50(IMAGERES)

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
	dataGen = readDFW.getGenerator(normGen, normImpGen, impGenNorm, FLAGS.batch_size, 0)

	def dummy_fn():
		print("Creation")
		return model.siamese_net

	# SKLearn wrapper around model
	wrapped_model = KerasClassifier(dummy_fn)
	# wrapped_model.model = dummy_fn()

	# initialize ActiveLearner
	learner = ActiveLearner(
		estimator=wrapped_model,
		query_strategy=uncertainty_sampling,
		verbose=1)

	n_queries = 0
	while True:
		(X_old_left, X_old_right), Y_old = next(dataGen)
		if X_old_left is None:
			print("Done with all data")
			break
		query_idx, query_instance = learner.query([X_old_left, X_old_right],
													n_instances=int(len(X_old_left) * FLAGS.active_ratio),
													verbose=0)
		learner.teach(
			X=[X_old_left[query_idx], X_old_right[query_idx]],
			y=Y_old[query_idx],
			only_new=True,
			verbose=1
		)
		n_queries += len(query_idx)
		print("Query no : %d" % n_queries)

	# Save model
	model.siamese_net.save_weights(FLAGS.out_model + ".h5")
	print("Queries made  : %d" % (n_queries))
