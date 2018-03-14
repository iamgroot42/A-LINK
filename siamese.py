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
	
	def trainModel(self, trainDatagen, epochs, batch_size, verbose=1):
		early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=5, verbose=verbose)
		reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.01, verbose=verbose)
		self.siamese_net.fit_generator(trainDatagen
			,steps_per_epoch=400000 / batch_size, epochs=epochs
			,callbacks=[early_stop, reduce_lr])
	
	def testAccuracy(self, X, Y):
		n_correct = 0
		for i in range(len(X)):
			for j in range(len(X)):
				for x in X[i]:
					for y in X[j]:
						prediction = np.argmax(self.predict([x, y]))
						if prediction == 1 and Y[i] == Y[j]:
							n_correct += 1
						elif prediction == 0 and Y[i] != Y[j]:
							n_correct += 1
		return n_correct / float( len(X) ** 2)

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
		return self.siamese_net.predict(self.preprocess(X))


class FaceVGG16(CustomModel, object):
	def __init__(self, shape, modelName, learningRate=1.0)
		super(FaceVGG16, self).__init__(shape, modelName, learningRate)
		vgg_model = VGGFace(model='vgg16', include_top=False, input_shape=self.shape)
		last_layer = vgg_model.get_layer('pool5').output
		out = Flatten(name='flatten')(last_layer)
		self.baseModel = Model(vgg_model.input, out)

	def preprocess(self, X):
		X_temp = np.copy(X)
		return utils.preprocess_input(X_temp, version=1)


class RESNET50(CustomModel, object):
	def __init__(self, shape, modelName, learningRate=1.0):
		super(RESNET50, self).__init__(shape, modelName, learningRate)
		vgg_model = VGGFace(model='resnet50', include_top=False, input_shape=self.shape)
		last_layer = vgg_model.get_layer('avg_pool').output
		out = Flatten(name='flatten')(last_layer)
		self.baseModel = Model(vgg_model.input, out)

	def preprocess(self, X):
		X_temp = np.copy(X)
		return utils.preprocess_input(X_temp, version=2)
