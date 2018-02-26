import numpy as np
from PIL import Image
from scipy.misc import imresize
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import os

# Set seed for reproducability
np.random.seed(42)

# Get count of test data
def getContentsSize(imagePaths):
	return len(os.listdir(imagePaths))

# Generator for reading through directory
def directoryGenerator(preprofunc = None):
	datagen = ImageDataGenerator(
	    	width_shift_range=0.15,
	    	height_shift_range=0.15,
	    	horizontal_flip=True,
	    	preprocessing_function=preprofunc)
	return datagen

# Get generators for train and val data
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

# Resize images
def resize(images, new_size):
	resized_images = []
	for image in images:
		resized_images.append(imresize(image, new_size))
	resized_images = np.array(resized_images)
	resized_images = resized_images.astype("float32")
	return np.array(resized_images)

# Test data generator
def testDataGenerator(baseDir, imagePaths, highRes, lowRes, batch_size=128):
	i = 0
	X_low = []
	X_high = []
	Y = []
	with open(imagePaths, 'r') as f:
		for path in f:
				properPath = os.path.join(baseDir, path.rstrip('\n'))
				image = np.asarray(Image.open(properPath), dtype="int32")
				X_low.append(imresize(image, lowRes))
				X_high.append(imresize(image, highRes))
				Y.append(path.split('_')[0])
				i += 1
				if i == batch_size:
					yield np.array(X_low), np.array(X_high), np.array(Y)
					i = 0
					X_low = []
					X_high = []
					Y = []

# Load data from memory, resized into desired shape
def resizedLoadData(baseDir, imagesFolder, desiredRes):
	X = []
	Y = []
	for classLabel in os.listdir(imagesFolder):
		subDirPath = os.path.join(imagesFolder, classLabel)
		for path in os.listdir(subDirPath):
			properPath = os.path.join(subDirPath, path)
			image = np.asarray(Image.open(properPath), dtype="int32")
			X.append(imresize(image, desiredRes))
			Y.append(path.split('_')[0])
	return np.array(X), Y

# Get generator for unlabelled data
def getUnlabelledData(baseDir, imagePaths, batch_size=32):
		i = 0
		X = []
		Y = []
		with open(imagePaths, 'r') as f:
			for path in f:
				properPath = os.path.join(baseDir, path.rstrip('\n'))
				image = np.asarray(Image.open(properPath), dtype="int32")
				X.append(image)
				Y.append(path.split('_')[0])
				i += 1
				if i ==  batch_size:
					yield np.array(X), np.array(Y)
					i = 0
					X = []
					Y = []

# Load train, val data from folders
def resizeLoadDataAll(baseDir, trainImagePaths, valImagePaths, desiredRes):
	X_train, Y_train = resizedLoadData(baseDir, trainImagePaths, desiredRes)
	X_val, Y_val = resizedLoadData(baseDir, valImagePaths, desiredRes)
	uniqueClasses = list(set(Y_train))
	classMapping = {y:x for x,y in enumerate(uniqueClasses)}
	Y_train = np_utils.to_categorical([classMapping[x] for x in Y_train], len(uniqueClasses))
	Y_val = np_utils.to_categorical([classMapping[x] for x in Y_val], len(uniqueClasses))
	return (X_train, Y_train), (X_val, Y_val), classMapping
