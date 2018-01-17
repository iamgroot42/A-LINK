import numpy as np
from PIL import Image
from scipy.misc import imresize
from keras.preprocessing.image import ImageDataGenerator

import os

# Set seed for reproducability
np.random.seed(42)

def directoryGenerator(preprofunc = None):
	datagen = ImageDataGenerator(
	    	width_shift_range=0.15,
	    	height_shift_range=0.15,
	    	horizontal_flip=True,
	    	preprocessing_function=preprofunc)
	return datagen


def returnGenerators(train_dir, val_dir, imageSize, batchSize, preprofunc=None):
	train_datagen = directoryGenerator(preprofunc)
	train_generator = train_datagen.flow_from_directory(
    		train_dir,
	    	target_size=imageSize,
	    	batch_size=batchSize)
	test_datagen = directoryGenerator(preprofunc)
	validation_generator = test_datagen.flow_from_directory(
	    	val_dir,
	    	target_size=imageSize,
	    	batch_size=batchSize)
	return train_generator, validation_generator


def resize(images, new_size):
	resized_images = []
	for image in images:
		resized_images.append(imresize(image, new_size))
	resized_images = np.array(resized_images)
	resized_images = resized_images.astype("float32")
	return np.array(resized_images)


def loadTestData(baseDir, imagePaths, highRes, lowRes):
	X = []
	Y = []
	with open(imagePaths, 'r') as f:
		for path in f:
			properPath = os.path.join(baseDir, path.rstrip('\n'))
			image = np.asarray(Image.open(properPath), dtype="int32")
			X.append(image)
			Y.append(int(path.split('_')[0])-1)
	X_lr = resize(X, lowRes)
	X_hr = resize(X, highRes)
	return X_lr, X_hr, np.array(Y)


def getUnlabelledData(baseDir, imagePaths, batch_size=32):
		i = 0
		X = []
		Y = []
		with open(imagePaths, 'r') as f:
			for path in f:
				properPath = os.path.join(baseDir, path.rstrip('\n'))
				image = np.asarray(Image.open(properPath), dtype="int32")
				X.append(image)
				Y.append(int(path.split('_')[0])-1)
				i += 1
				if i ==  batch_size:
					yield np.array(X), np.array(Y)
					i = 0
					X = []
					Y = []
