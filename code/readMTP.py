import os
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm


def readAllImages(dirPath):
	person_wise = {}
	people = []
	for path in os.listdir(dirPath):
		person_id = int(path.split('_')[0])
		if "_051_06" in path or "_051_08" in path:
			if person_id not in person_wise:
				person_wise[person_id] = []
			person_wise[person_id].append(path)
	person_keys = person_wise.keys()
	for key in tqdm(person_keys):
		temp_people = []
		for person in person_wise[key]:
			img = cv2.resize(np.asarray(Image.open(os.path.join(dirPath, path)).convert('RGB'), dtype=np.float32), (150, 150))
			temp_people.append(img)
		people.append(np.stack(temp_people))
	return people


def generatorFeaturized(X, Y, batch_size, featurize=None, resize_res=None):
	X_left, X_right, Y_send = [], [], []
	while True:
		for i in range(0, len(X), batch_size):
			x_left  = np.array(X[0][i: i + batch_size])
			x_right = np.array(X[1][i: i + batch_size])
			Y       = np.array(Y[i: i + batch_size])
			# De-bias data
			Y_flat  = np.stack([y[0] for y in Y])
			pos = np.where(Y_flat == 1)[0]
			neg = np.where(Y_flat == 0)[0]
			# Don't train on totally biased data
			minSamp = np.minimum(len(pos), len(neg))
			if minSamp == 0:
				continue
			selectedIndices = np.concatenate((np.random.choice(pos, minSamp, replace=False), np.random.choice(neg, minSamp, replace=False)), axis=0)
			Y = Y[selectedIndices]
			x_left, x_right = x_left[selectedIndices], x_right[selectedIndices]
			# Resize, if asked to
			if resize_res:
				x_left, x_right = resizeImages([x_left, x_right], resize_res)
			# Featurize, if asked to
			if featurize:
				x_left  = featurize.process(x_left)
				x_right = featurize.process(x_right)
			if len(Y_send) > 0:
				X_left = np.concatenate((X_left, x_left), axis=0)
				X_right = np.concatenate((X_right, x_right), axis=0)
				Y_send = np.concatenate((Y_send, Y), axis=0)
			else:
				X_left = np.copy(x_left)
				X_right = np.copy(x_right)
				Y_send = np.copy(Y)
			if len(Y_send) >= batch_size:
				yield ([X_left, X_right], Y_send)
				X_left, X_right, Y_send = [], [], []


def getGenerator(datGen, batch_size, val_ratio=0.2, resize_res=None, featurize=None):
	X_left, X_right, Y_send = [], [], []
	while True:
		X, Y = datGen.next()
		# Generate data in 1:1 ratio to avoid overfitting
		Y_flat = np.stack([y[0] for y in Y])
		pos = np.where(Y_flat == 1)[0]
		neg = np.where(Y_flat == 0)[0]
		minSamp = np.minimum(len(pos), len(neg))
		# Don't train on totally biased data
		if minSamp == 0:
			continue
		selectedIndices = np.concatenate((np.random.choice(pos, minSamp, replace=False), np.random.choice(neg, minSamp, replace=False)), axis=0)
		Y = Y[selectedIndices]
		X = [X[0][selectedIndices], X[1][selectedIndices]]
		# Resize, if asked to
		if resize_res:
			X[0], X[1] = resizeImages([X[0], X[1]], resize_res)
		# Featurize, if asked to
		if featurize:
			X[0]  = featurize.process(X[0])
			X[1] = featurize.process(X[1])
		if len(Y_send) > 0:
			X_left = np.concatenate((X_left, X[0]), axis=0)
			X_right = np.concatenate((X_right, X[1]), axis=0)
			Y_send = np.concatenate((Y_send, Y), axis=0)
		else:
			X_left = np.copy(X[0])
			X_right = np.copy(X[1])
			Y_send = np.copy(Y)
		if len(Y_send) >= batch_size:
			yield ( [X_left, X_right], Y_send)
			X_left, X_right, Y_send = [], [], []


def resizeImages(images, resize_res):
	resized_left  = [cv2.resize(image, resize_res) for image in images[0]]
	resized_right = [cv2.resize(image, resize_res) for image in images[1]]
	return [np.array(resized_left), np.array(resized_right)]
