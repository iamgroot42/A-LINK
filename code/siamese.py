from keras.engine import  Model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Input, Activation, Dropout, Conv2D, MaxPooling2D, Lambda
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from keras.optimizers import Adadelta, rmsprop
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import backend as K

import numpy as np
import sys


class SiameseNetwork:
	def __init__(self, shape, modelName, learningRate=1.0):
		self.learningRate = learningRate
		self.modelName = modelName
		left_input = Input(shape)
		right_input = Input(shape)
		# Define Siamese Network using shared weights
		L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
		L1_distance = L1_layer([left_input, right_input])
		hidden = Dense(512, activation='relu')(L1_distance)
		hidden2 = Dense(64, activation='relu')(hidden)
		prediction = Dense(1, activation='sigmoid')(hidden2)
		self.siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)
		# Compile and prepare network
		self.siamese_net.compile(loss="binary_crossentropy", optimizer=Adadelta(self.learningRate), metrics=['accuracy'])

	def trainModel(self, trainDatagen, valGen, epochs, batch_size, verbose=1):
		early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=5, verbose=verbose)
		reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.01, verbose=verbose)
		self.siamese_net.fit_generator(trainDatagen
			,steps_per_epoch=320000 / batch_size, epochs=epochs
			#,validation_data=valGen, validation_steps = 80000 / batch_size
			,callbacks=[early_stop, reduce_lr])

	def finetune(self, X, Y, epochs, batch_size, verbose=1):
		early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=5, verbose=1)
		reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.01, verbose=verbose)
		self.siamese_net.fit(self.preprocess(X), Y, batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=verbose, callbacks=[early_stop, reduce_lr])

	def testAccuracy(self, X, Y, batch_size=512):
		n_correct, total = 0, 0
		X_left, X_right, Y_send = [], [], []
		for i, x in enumerate(X):
			for j, y in enumerate(X):
				X_left.append(x)
				X_right.append(y)
				Y_send.append(1*(Y[i] == Y[j]))
				if len(X_left) == batch_size:
					Y_send = np.stack(Y_send)
					predictions = np.argmax(self.siamese_net.predict([np.stack(X_left), np.stack(X_right)]), axis=1)
					n_correct += np.sum(predictions == Y_send)
					total += len(X_left)
					X_left, X_right, Y_send = [], [], []
		if len(X_left) > 0:
			Y_send = np.stack(Y_send)
			predictions = np.argmax(self.siamese_net.predict([np.stack(X_left), np.stack(X_right)]), axis=1)
			n_correct += np.sum(predictions == Y_send)
			total += len(X_left)
		return n_correct / float(total)

	def customTrainModel(self, dataGen, epochs, batch_size, valRatio=0.2):
		steps_per_epoch = 320000 / batch_size
		for _ in range(epochs):
			train_loss, val_loss = 0, 0
	                train_acc, val_acc = 0, 0
			for i in range(steps_per_epoch):
				x, y = dataGen.next()
				# Split into train and val
				indices = np.random.permutation(len(y))
		                splitPoint = int(len(y) * valRatio)
				x_train, y_train = [ pp[indices[splitPoint:]] for pp in x], y[indices[splitPoint:]]
				x_test, y_test = [ pp[indices[:splitPoint]] for pp in x], y[indices[:splitPoint]]
				train_metrics = self.siamese_net.train_on_batch(x_train, y_train)
				train_loss += train_metrics[0]
				train_acc += train_metrics[1]
				val_metrics = self.siamese_net.test_on_batch(x_test, y_test)
				val_loss += val_metrics[0]
				val_acc += val_metrics[1]
				sys.stdout.write("%d / %d : Tr loss: %f, Tr acc: %f, Vl loss: %f, Vl acc: %f  \r" % (i+1, steps_per_epoch, train_loss/(i+1), train_acc/(i+1), val_loss/(i+1), val_acc/(i+1)))
				sys.stdout.flush()
			print("\n")

	def maybeLoadFromMemory(self):
		try:
			self.siamese_net.load_weights(self.modelName + ".h5")
			return True
		except:
			return False

	def save(self, customName=None):
		if not customName:
			self.siamese_net.save_weights(self.modelName + ".h5")
		else:
			self.siamese_net.save_weights(customName + ".h5")

	def preprocess(self, X):
		return X

	def predict(self, X):
		return self.siamese_net.predict(self.preprocess(X))


class SmallRes(SiameseNetwork, object):
	def __init__(self, imageShape, featureShape, name, learningRate):
		self.learningRate = learningRate
		self.modelName = name
		convnet = Sequential()
		convnet.add(Conv2D(32, (3, 3), padding='same',
					input_shape=imageShape))
		convnet.add(Activation('relu'))
		convnet.add(Conv2D(32, (3, 3)))
		convnet.add(Activation('relu'))
		convnet.add(MaxPooling2D(pool_size=(2, 2)))
		convnet.add(Dropout(0.25))

		convnet.add(Conv2D(64, (3, 3), padding='same'))
		convnet.add(Activation('relu'))
		convnet.add(Conv2D(64, (3, 3)))
		convnet.add(Activation('relu'))
		convnet.add(MaxPooling2D(pool_size=(2, 2)))
		convnet.add(Dropout(0.25))

		convnet.add(Flatten())
		convnet.add(Dense(featureShape[0]))
		convnet.add(Activation('relu'))

		left_input = Input(imageShape)
		right_input = Input(imageShape)
		encoded_l = convnet(left_input)
		encoded_r = convnet(right_input)
		L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
		L1_distance = L1_layer([encoded_l, encoded_r])
		hidden = Dense(512, activation='relu')(L1_distance)
		hidden2 = Dense(64, activation='relu')(hidden)
		prediction = Dense(1, activation='sigmoid')(hidden2)
		self.siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)
		# Compile and prepare network
		self.siamese_net.compile(loss="binary_crossentropy", optimizer=Adadelta(self.learningRate), metrics=['accuracy'])


class FaceVGG16:
	def __init__(self, shape):
		vgg_model = VGGFace(model='vgg16', include_top=False, input_shape=shape + (3,))
		last_layer = vgg_model.get_layer('pool5').output
		out = Flatten(name='flatten')(last_layer)
		self.model = Model(vgg_model.input, out)
		print self.model.output_shape

	def preprocess(self, X):
		X_temp = np.copy(X)
		return utils.preprocess_input(X_temp, version=1)

	def process(self, X):
		return self.model.predict(self.preprocess(X))


class RESNET50:
	def __init__(self, shape):
		vgg_model = VGGFace(model='resnet50', include_top=False, input_shape=shape + (3,))
		last_layer = vgg_model.get_layer('avg_pool').output
		out = Flatten(name='flatten')(last_layer)
		self.model = Model(vgg_model.input, out)

	def preprocess(self, X):
		X_temp = np.copy(X)
		return utils.preprocess_input(X_temp, version=2)

	def process(self, X):
		return self.model.predict(self.preprocess(X))
