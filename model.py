from keras.engine import  Model
from keras.layers import Flatten, Dense, Input
from keras_vggface.vggface import VGGFace
from keras_vggface import utils

from keras.optimizers import Adadelta, SGD, rmsprop
import cv2


class Model:
	def __init__(self, shape, out_dim):
		self.shape = shape
		pass

	def finetune(self, X, Y, epochs, learning_rate, batch_size):
		# Add implementation
		pass

	def predict(self, X):
		# Add implementation
		pass


class VGG16(Model):
	def __init__(self, shape, out_dim, hid_dim):
		self.hid_dim = hid_dim
		super(VGG16, self).__init__(shape, out_dim)
		vgg_notop = VGGFace(model='vgg16', include_top=False, input_shape=self.shape)
		last_layer = vgg_model.get_layer('pool5').output
		x = Flatten(name='flatten')(last_layer)
		x = Dense(self.hid_dim, activation='relu', name='fc6')(x)
		x = Dense(self.hid_dim, activation='relu', name='fc7')(x)
		out = Dense(self.out_dim, activation='softmax', name='fc8')(x)
		self.model = Model(vgg_model.input, out)
		self.model.compile(optimizer=Adadelta(lr=learning_rate), metrics=['accuracy'])

	def finetune(self, X, Y, epochs, learning_rate, batch_size):
		self.model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_split=0.2)

	def predict(self, X):
		x_tr = utils.preprocess_input(X, version=1)
		preds = self.model.predict(x_tr)
		return utils.decode_predictions(preds)


class RESNET50(Model):
	def __init__(self, shape, out_dim):
		super(RESNET50, self).__init__(shape, out_dim)
		vgg_notop = VGGFace(model='resnet50', include_top=False, input_shape=self.shape)
		last_layer = vgg_model.get_layer('avg_pool').output
		x = Flatten(name='flatten')(last_layer)
		out = Dense(self.out_dim, activation='softmax', name='classifier')(x)
		self.model = Model(vgg_model.input, out)
		self.model.compile(optimizer=Adadelta(lr=learning_rate), metrics=['accuracy'])

	def finetune(self, X, Y, epochs, learning_rate, batch_size):
		self.model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_split=0.2)

	def predict(self, X):
		x_tr = utils.preprocess_input(X, version=2)
		preds = self.model.predict(x_tr)
		return utils.decode_predictions(preds)


class SENET50(Model):
	def __init__(self, shape, out_dim):
		super(SENET50, self).__init__(shape, out_dim)
		vgg_notop = VGGFace(model='senet50', include_top=False, input_shape=self.shape)
		last_layer = vgg_model.get_layer('avg_pool').output
		x = Flatten(name='flatten')(last_layer)
		out = Dense(self.out_dim, activation='softmax', name='classifier')(x)
		self.model = Model(vgg_model.input, out)
		self.model.compile(optimizer=Adadelta(lr=learning_rate), metrics=['accuracy'])

	def finetune(self, X, Y, epochs, learning_rate, batch_size):
		self.model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_split=0.2)

	def predict(self, X):
		x_tr = utils.preprocess_input(X, version=2)
		preds = self.model.predict(x_tr)
		return utils.decode_predictions(preds)


def SmallRes(Model):
	def __init__(self, shape, out_dim):
		super(SmallRes, self).__init__(shape, out_dim)
		self.model = Sequential()
		self.model.add(Conv2D(32, (3, 3), padding='same',
                 	input_shape=shape))
		self.model.add(Activation('relu'))
		self.model.add(Conv2D(32, (3, 3)))
		self.model.add(Activation('relu'))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Dropout(0.25))

		self.model.add(Conv2D(64, (3, 3), padding='same'))
		self.model.add(Activation('relu'))
		self.model.add(Conv2D(64, (3, 3)))
		self.model.add(Activation('relu'))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Dropout(0.25))

		self.model.add(Flatten())
		self.model.add(Dense(512))
		self.model.add(Activation('relu'))
		self.model.add(Dropout(0.5))
		self.model.add(Dense(out_dim))
		self.model.add(Activation('softmax'))
		self.model.compile(rmsprop(lr=learning_rate, decay=1e-6, metrics=['accuracy']))

	def finetune(self, X, Y, epochs, learning_rate, batch_size):
		self.model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_split=0.2)

	def predict(self, X):
		return self.model.predict(X)
