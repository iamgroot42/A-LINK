from keras.losses import categorical_crossentropy
from keras.engine import  Model
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Input, Activation, Dropout, Conv2D, MaxPooling2D
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from keras.optimizers import Adadelta, SGD, rmsprop
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

from sklearn.utils import shuffle

import numpy as np
import cv2
import helpers


class CustomModel:
	def __init__(self, shape, modelName, learningRate=1.0):
		self.shape = shape + (3,)
		self.baseModel = None
		self.modelName = modelName
		self.left_input = Input(self.shape)
		self.right_input = Input(self.shape)

	def initializeModel(self):
		# Use feature extraction as it is
		for i in range(len(self.baseModel.layers)):
			self.baseModel.layers[i].trainable = False
		# Define Siamese Network using shared weights
		encoded_l = self.baseModel(self.left_input)
		encoded_r = self.baseModel(self.right_input)
		L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
		L1_distance = L1_layer([encoded_l, encoded_r])
		prediction = Dense(1, activation='sigmoid')(L1_distance)
		self.siamese_net = Model(inputs=[self.left_input, self.right_input], outputs=prediction)
		# Compile and prepare network
		self.siamese_net.compile(loss="binary_crossentropy", optimizer=Adadelta(learningRate))
	
	def trainModel(self, X_train, X_val, epochs, batch_size, verbose=1):
		# Expects data shape as N_classes x samples_per_class x 105 x 105 x 3
		n_classes = X_train.shape[0]
		n_examples = X_train.shape[1]
		assert(n_classes == X_val.shape[0])

		def train_batchgen(data):
			while True:
				categories = np.random.choice(n_classes, size=(batch_size,), replace=False)
				pairs = [np.zeros((batch_size,) + self.shape) for i in range(2)]
				targets = np.zeros((batch_size,))
				targets[batch_size//2:] = 1
				for i in range(batch_size):
					category = categories[i]
					idx_1 = np.random.randint(0, n_examples)
					pairs[0][i, :, :, :] = self.preprocess(data[category,idx_1].reshape(self.shape))
					idx_2 = np.random.randint(0, n_examples)
					#pick images of same class for 1st half, different for 2nd
					category_2 = category if i >= batch_size//2 else (category + np.random.randint(1, n_classes)) % n_classes
					pairs[1][i, :, :, :] = self.preprocess(data[category_2,idx_2].reshape(self.shape))
				yield pairs, targets

		trainDatagen = batchgen(X_train)
		valData = batchgen(X_val)

		early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=5, verbose=1)
		reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.01)
		self.siamese_net.fit_generator(trainDatagen
			,validation_data=valData
			,validation_steps=len(X_val) / batch_size
			,steps_per_epoch=len(X_train) / batch_size, epochs=epochs
			,callbacks=[early_stop, reduce_lr])

	def make_oneshot_task(self, X, N):
		n_val = X.shape[0]
		n_ex_val = X.shape[1]
		categories = np.random.choice(n_val, size=(N,), replace=False)
		indices = np.random.randint(0,self.n_ex_val, size=(N,))
		true_category = categories[0]
		ex1, ex2 = np.random.choice(self.n_examples, replace=False, size=(2,))
		test_image = np.asarray([X[true_category,ex1, :, :]] * N).reshape((N,) + self.shape)
		support_set = X[categories,indices, :, :]
		support_set[0, :, :] = X[true_category,ex2]
		support_set = support_set.reshape((N,) + self.shape)
		pairs = [test_image, support_set]
		targets = np.zeros((N,))
		targets[0] = 1
		return pairs, targets

	def make_oneshot_task(self, X, N):
		n_classes, n_examples = X.shape[:2]
		indices = np.random.randint(0, n_examples, size=(N,))
		categories = np.random.choice(range(n_classes), size=(N,), replace=False)            
		true_category = categories[0]
		ex1, ex2 = np.random.choice(n_examples, replace=False, size=(2,))
		test_image = np.asarray([X[true_category,ex1, :, :]] * N).reshape((N,) + self.shape)
		support_set = X[categories,indices, :, :]
		support_set[0, :, :] = X[true_category, ex2]
		support_set = support_set.reshape((N,) + self.shape)
		targets = np.zeros((N,))
		targets[0] = 1
		targets, test_image, support_set = shuffle(targets, test_image, support_set)
		pairs = [test_image,support_set]

		return pairs, targets
	
	def testAccuracy(self, X, N, k):
		n_correct = 0
		 for i in range(k):
			inputs, targets = self.make_oneshot_task(self.preprocess(X), N)
			probs = self.siamese_net.predict(inputs)
			if np.argmax(probs) == np.argmax(targets):
				n_correct += 1
		percent_correct = n_correct / float(k)
	   return percent_correct

	def maybeLoadFromMemory(self):
		try:
			self.siamese_net.load_weights(self.modelName + ".h5")
			return True
		except:
			return False

	def save(self):
		self.siamese_net.save_weights(self.modelName + ".h5")

	def preprocess(self, X):
		return X

	def predict(self, X):
		pass


# class FaceVGG16(CustomModel, object):
# 	def __init__(self, shape, out_dim, hid_dim):
# 		self.hid_dim = hid_dim
# 		super(FaceVGG16, self).__init__(shape, out_dim, "FaceVGG16")
# 		vgg_model = VGGFace(model='vgg16', include_top=False, input_shape=self.shape)
# 		last_layer = vgg_model.get_layer('pool5').output
# 		x = Flatten(name='flatten')(last_layer)
# 		x = Dense(self.hid_dim, activation='relu', name='fc6')(x)
# 		x = Dense(self.hid_dim, activation='relu', name='fc7')(x)
# 		out = Dense(self.out_dim, activation='softmax', name='fc8')(x)
# 		self.model = Model(vgg_model.input, out)(x)

# 	def preprocess(self, X):
# 		X_temp = np.copy(X)
# 		return utils.preprocess_input(X_temp, version=1)

# 	def predict(self, X):
# 		preds = self.model.predict(self.preprocess(X))
# 		return preds


# class RESNET50(CustomModel, object):
# 	def __init__(self, shape, out_dim):
# 		super(RESNET50, self).__init__(shape, out_dim, "RESNET50")
# 		vgg_model = VGGFace(model='resnet50', include_top=False, input_shape=self.shape)
# 		last_layer = vgg_model.get_layer('avg_pool').output
# 		x = Flatten(name='flatten')(last_layer)
# 		out = Dense(self.out_dim, activation='softmax', name='classifier')(x)
# 		self.model = Model(vgg_model.input, out)
# 		self.model.compile(loss=categorical_crossentropy,
# 			optimizer=Adadelta(lr=1.0), metrics=['accuracy'])

# 	def preprocess(self, X):
# 		X_temp = np.copy(X)
# 		return utils.preprocess_input(X_temp, version=2)

# 	def predict(self, X):
# 		preds = self.model.predict(self.preprocess(X))
# 		return preds


# class SENET50(CustomModel, object):
# 	def __init__(self, shape, out_dim):
# 		super(SENET50, self).__init__(shape, out_dim, "SETNET50")
# 		vgg_model = VGGFace(model='senet50', include_top=False, input_shape=self.shape)
# 		last_layer = vgg_model.get_layer('avg_pool').output
# 		x = Flatten(name='flatten')(last_layer)
# 		out = Dense(self.out_dim, activation='softmax', name='classifier')(x)
# 		self.model = Model(vgg_model.input, out)

# 	def preprocess(self, X):
# 		X_temp = np.copy(X)
# 		return utils.preprocess_input(X_temp, version=2)

# 	def predict(self, X):
# 		preds = self.model.predict(self.preprocess(X))
# 		return preds
