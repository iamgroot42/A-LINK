from keras.losses import categorical_crossentropy
from keras.engine import  Model
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Input, Activation, Dropout, Conv2D, MaxPooling2D
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from keras.optimizers import Adadelta, SGD, rmsprop
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import cv2
import helpers


class CustomModel:
	def __init__(self, shape, out_dim, modelName):
		self.shape = shape + (3,)
		self.out_dim = out_dim
		self.modelName = modelName

	def maybeLoadFromMemory(self):
		try:
			self.model = load_model(self.modelName)
			return True
		except:
			return False

	def save(self):
		self.model.save(self.modelName)

	def preprocess(self, X):
		return X

	def finetune(self, X, Y, epochs, batch_size, verbose=1):
		early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=5, verbose=1)
		self.model.fit(self.preprocess(X), Y, batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=verbose, callbacks=[early_stop])

	def trainModel(self, X_train, Y_train, X_val, Y_val, epochs, batch_size, verbose=1):
		early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=5, verbose=1)
		self.model.fit(self.preprocess(X_train), Y_train, validation_data=(self.preprocess(X_val), Y_val) ,batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=[early_stop])

	def trainWithAugmentation(self, X, Y, epochs, batch_size, verbose=1):
		(X_train, Y_train), (X_val, Y_val) = helpers.unisonSplit(X, Y, 0.8)
		trainDatagen = ImageDataGenerator(rotation_range=10,
			width_shift_range=0.1,
			height_shift_range=0.1,
			shear_range=10,
			horizontal_flip=True,
			preprocessing_function=self.preprocess)
		trainDatagen.fit(X_train)
		valDatagen = ImageDataGenerator(rotation_range=10,
			width_shift_range=0.1,
			height_shift_range=0.1,
			shear_range=10,
			horizontal_flip=True,
			preprocessing_function=self.preprocess)
		valDatagen.fit(X_val)
		self.model.fit_generator(trainDatagen.flow(X_train, Y_train, batch_size=batch_size),
			validation_data=valDatagen.flow(X_val, Y_val, batch_size=batch_size),
			validation_steps=len(X_val) / batch_size,
			steps_per_epoch=len(X_train) / batch_size, epochs=epochs)

	def trainWithoutVal(self, X, Y, epochs, batch_size, verbose=1):
		modifiedX = self.preprocess(X)
		early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=5, verbose=1)
		self.model.fit(modifiedX, Y, validation_split=0.2, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=[early_stop])

	def finetuneGenerator(self, trainGen, valGen, steps_epoch, batch_size, epochs, verbose=1):
		self.model.fit_generator(
			trainGen,
			steps_per_epoch=steps_epoch // batch_size,
			epochs=epochs,
			verbose=verbose,
			validation_data=valGen,
			validation_steps=800 // batch_size)

	def predict(self, X):
		pass

	def finetuneDenseOnly(self, X, Y, epochs, batch_size, verbose=1):
		early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=5, verbose=1)
		self.model.fit(self.preprocess(X), Y, batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=verbose, callbacks=[early_stop])


class FaceVGG16(CustomModel, object):
	def __init__(self, shape, out_dim, hid_dim):
		self.hid_dim = hid_dim
		super(FaceVGG16, self).__init__(shape, out_dim, "FaceVGG16")
		vgg_model = VGGFace(model='vgg16', include_top=False, input_shape=self.shape)
		last_layer = vgg_model.get_layer('pool5').output
		x = Flatten(name='flatten')(last_layer)
		x = Dense(self.hid_dim, activation='relu', name='fc6')(x)
		x = Dense(self.hid_dim, activation='relu', name='fc7')(x)
		out = Dense(self.out_dim, activation='softmax', name='fc8')(x)
		self.model = Model(vgg_model.input, out)(x)

	def preprocess(self, X):
		X_temp = np.copy(X)
		return utils.preprocess_input(X_temp, version=1)

	def predict(self, X):
		preds = self.model.predict(self.preprocess(X))
		return preds


class RESNET50(CustomModel, object):
	def __init__(self, shape, out_dim):
		super(RESNET50, self).__init__(shape, out_dim, "RESNET50")
		vgg_model = VGGFace(model='resnet50', include_top=False, input_shape=self.shape)
		last_layer = vgg_model.get_layer('avg_pool').output
		x = Flatten(name='flatten')(last_layer)
		out = Dense(self.out_dim, activation='softmax', name='classifier')(x)
		self.model = Model(vgg_model.input, out)
		self.model.compile(loss=categorical_crossentropy,
			optimizer=Adadelta(lr=1.0), metrics=['accuracy'])

	def preprocess(self, X):
		X_temp = np.copy(X)
		return utils.preprocess_input(X_temp, version=2)

	def predict(self, X):
		preds = self.model.predict(self.preprocess(X))
		return preds


class SENET50(CustomModel, object):
	def __init__(self, shape, out_dim):
		super(SENET50, self).__init__(shape, out_dim, "SETNET50")
		vgg_model = VGGFace(model='senet50', include_top=False, input_shape=self.shape)
		last_layer = vgg_model.get_layer('avg_pool').output
		x = Flatten(name='flatten')(last_layer)
		out = Dense(self.out_dim, activation='softmax', name='classifier')(x)
		self.model = Model(vgg_model.input, out)

	def preprocess(self, X):
		X_temp = np.copy(X)
		return utils.preprocess_input(X_temp, version=2)

	def predict(self, X):
		preds = self.model.predict(self.preprocess(X))
		return preds


class SmallRes(CustomModel, object):
	def __init__(self, shape, out_dim):
		super(SmallRes, self).__init__(shape, out_dim, "SmallRes")
		self.model = Sequential()
		self.model.add(Conv2D(32, (3, 3), padding='same',
					input_shape=self.shape))
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
		self.model.compile(loss=categorical_crossentropy,
			optimizer=rmsprop(lr=0.001, decay=1e-6), metrics=['accuracy'])

	def preprocess(self, X):
		return X / 255.0

	def predict(self, X):
		return self.model.predict(self.preprocess(X))
