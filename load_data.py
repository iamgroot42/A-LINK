import numpy as np
from PIL import Image
from scipy.misc import imresize
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import os

# Set seed for reproducability
np.random.seed(42)

# Get count of test data
def getContentsSize(imagesPath):
	with open(imagesPath, 'r') as f:
		for i, _ in enumerate(f):
			pass
	return i + 1

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
def getUnlabelledData(baseDir, imagePaths, batch_size=8):
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

# Create pairs from images with person labels
def labelToSiamese(X, Y):
	X_left, X_right, Y_ = [], [], []
	for i in range(len(Y)):
		for j in range(i, len(Y)):
			X_left.append(X[i])
			X_right.append(X[j])
			if Y[i] == Y[j]:
				Y_.append([1])
			else:
				Y_.append([0])
	return [np.stack(X_left), np.stack(X_right)], Y_

# Load train, val data from folders
def resizeLoadDataAll(baseDir, trainImagePaths, valImagePaths, desiredRes):
	X_train, Y_train = resizedLoadData(baseDir, trainImagePaths, desiredRes)
	X_val, Y_val = resizedLoadData(baseDir, valImagePaths, desiredRes)
	uniqueClasses = list(set(Y_train))
	classMapping = {y:x for x,y in enumerate(uniqueClasses)}
	Y_train = np_utils.to_categorical([classMapping[x] for x in Y_train], len(uniqueClasses))
	Y_val = np_utils.to_categorical([classMapping[x] for x in Y_val], len(uniqueClasses))
	X_train = np.concatenate((X_train, X_val))
	Y_train = np.concatenate((Y_train, Y_val))
	return (X_train, Y_train)

# Create generator from given data with class labels
def dataToSiamGen(X, Y, batch_size):
	while True:
		for i in range(0, len(Y), batch_size):
			yield X[i: i + batch_size], Y[i: i + batch_size]

# Combine data from train and val generators into a siamesestyle generator
def combineGenSiam(gen1, gen2, conversionModel, batch_size):
	X_left, X_right, Y_send = [], [], []
	while True:
		X1, Y1 = gen1.next()
		if gen2:
			X2, Y2 = gen2.next()
			X1, X2 = conversionModel.process(X1), conversionModel.process(X2)
			X, Y = np.concatenate((X1, X2), axis=0), np.concatenate((Y1, Y2), axis=0)
		else:
			X, Y = X1, Y1
		X_new_L, X_new_R, Y_ = [], [], []
		for i in range(len(Y)):
			for j in range(i, len(Y)):
				X_new_L.append(X[i])
				X_new_R.append(X[j])
				if np.argmax(Y[i]) == np.argmax(Y[j]):
					Y_.append([1])
				else:
					Y_.append([0])
		Y_flat = np.stack([y[0] for y in Y_])
		pos = np.where(Y_flat == 1)[0]
		neg = np.where(Y_flat == 0)[0]
		minSamp = np.minimum(len(pos), len(neg))
		# Don't train on totally biased data
		if minSamp == 0:
			continue
		selectedIndices = np.concatenate((np.random.choice(pos, minSamp, replace=False), np.random.choice(neg, minSamp, replace=False)), axis=0)
		X_new_L, X_new_R, Y_ = np.stack(X_new_L), np.stack(X_new_R), np.stack(Y_)
		Y_wow = Y_[selectedIndices]
		X_wow = [X_new_L[selectedIndices], X_new_R[selectedIndices]]
		if len(Y_send) > 0:
			X_left = np.concatenate((X_left, X_wow[0]), axis=0)
			X_right = np.concatenate((X_right, X_wow[1]), axis=0)
			Y_send = np.concatenate((Y_send, Y_wow), axis=0)
		else:
			X_left = np.copy(X_wow[0])
			X_right = np.copy(X_wow[1])
			Y_send = np.copy(Y_wow)
		if len(Y_send) >= batch_size:
			yield ( [X_left, X_right], Y_send)
			X_left, X_right, Y_send = [], [], []

# Read test-data
def testDataGenerator(baseDir, imagePaths, imageRes, batch_size=128):
	i = 0
	X = []
	Y = []
	with open(imagePaths, 'r') as f:
		for path in f:
			properPath = os.path.join(baseDir, path.rstrip('\n'))
			image = np.asarray(Image.open(properPath), dtype="int32")
			X.append(imresize(image, imageRes))
			Y.append(path.split('_')[0])
			i += 1
			if i == batch_size:
				yield np.array(X, dtype="float32"),  np.array(Y)
				i = 0
				X = []
				Y = []
