import numpy as np
from PIL import Image
import os
import cv2
import re


def lookupFile(fullPath):
		stupidString = '\xef\xbb\xbf'
		directory, fileName = fullPath.rsplit('/', 1)
		modifiedName, extension = fileName.rsplit('.', 1)
		if os.path.exists(fullPath):
				return fullPath
		elif os.path.exists(os.path.join(directory + stupidString, modifiedName) + "." + extension):
				return os.path.join(directory + stupidString, modifiedName) + "." + extension
		elif os.path.exists(os.path.join(directory + stupidString, modifiedName + stupidString) + "." + extension):
				return os.path.join(directory + stupidString, modifiedName + stupidString) + "." + extension
		elif os.path.exists(os.path.join(directory, modifiedName + stupidString) + "." + extension):
				return os.path.join(directory, modifiedName + stupidString) + "." + extension
		elif os.path.exists(os.path.join(directory, " " + modifiedName) + "." + extension):
				return os.path.join(directory, " " + modifiedName) + "." + extension
		else:
				print(fullPath)
				print(os.listdir(directory))
				return None


def cropImages(prefix, dirPath, faceBoxes):
	imagesBefore = sorted(os.listdir(os.path.join(prefix, dirPath)))
	i = 0
	for imPath in imagesBefore:
		try:
			partialName = os.path.join(dirPath, imPath)
			fullName = os.path.join(prefix, partialName)
			fullName =  lookupFile(re.sub(r"[/]\s", "/", fullName))
			img = Image.open(fullName).convert('RGB')
			tx, h, w, by = faceBoxes[partialName]
			img = img.crop((tx, h, w, by))
			img.save(fullName)
		except Exception:
			os.remove(fullName)
			i += 1
			print(e)
	return i


def constructIndexMap(filePath):
	with open(filePath, 'r') as f:
		mapping = {}
		for row in f:
			imgname, tx, h, w, by = row.rstrip('\n').rstrip().rsplit(' ', 4)
			mapping[imgname] = [float(x) for x in [tx, h, w, by]]
	return mapping


def cropAllFolders(prefix, trainFolder, boxMap):
	allImages = os.path.join(prefix, trainFolder)
	prob = 0
	personList = sorted(os.listdir(allImages))
	for person in personList:
		prob += cropImages(prefix, os.path.join(trainFolder, person), boxMap)
	print("Problem with", prob)


def getAllTrainData(prefix, trainFolder, imageRes, model, combine_normal_imp=False):
	allImages = os.path.join(prefix, trainFolder)
	X_plain = []
	X_dig = []
	X_imp = []
	personList = sorted(os.listdir(allImages))
	for person in personList:
		X_person_dig = []
		X_person_imp = []
		X_person_normal = []
		dirPath = os.path.join(allImages, person)
		faceList = sorted(os.listdir(dirPath))
		for impath in faceList:
			fullName = os.path.join(dirPath, impath)
			fullName = re.sub(r"[/]\s", "/", fullName)
			fileName = impath.rsplit('.', 1)[0]
			try:
				img = cv2.resize(np.asarray(Image.open(lookupFile(fullName)).convert('RGB'), dtype=np.float32), imageRes)
				if img.shape[0] !=224 or img.shape[1] != 224 or img.shape[2] !=3:
					print(img.shape)
				if '_h_' in fileName:
					if combine_normal_imp:
						X_person_normal.append(img) 
					else:
						X_person_dig.append(img)
				elif '_I_' in fileName:
					X_person_imp.append(img)
				else:
					X_person_normal.append(img)
			except Exception as ex:
				print(ex)
		if X_person_dig and X_person_imp and X_person_normal:
			if not combine_normal_imp:
				X_dig.append(model.process(np.stack(X_person_dig)))
			X_imp.append(model.process(np.stack(X_person_imp)))
			X_plain.append(model.process(np.stack(X_person_normal)))
	if not combine_normal_imp:
		 # 1 validation & 1 training sample per person
		assert(len(X_plain) == len(X_dig) and len(X_dig) == len(X_imp))
	return (X_plain, X_dig, X_imp)


def getRawTrainData(prefix, trainFolder, imageRes):
	allImages = os.path.join(prefix, trainFolder)
	X_plain = []
	X_dig = []
	personList = sorted(os.listdir(allImages))
	for person in personList:
		X_person_dig = []
		X_person_imp = []
		X_person_normal = []
		dirPath = os.path.join(allImages, person)
		faceList = sorted(os.listdir(dirPath))
		for impath in faceList:
			fullName = os.path.join(dirPath, impath)
			fullName = re.sub(r"[/]\s", "/", fullName)
			fileName = impath.rsplit('.', 1)[0]
			try:
				img = cv2.resize(np.asarray(Image.open(lookupFile(fullName)).convert('RGB'), dtype=np.float32), imageRes)
				if img.shape[0] !=224 or img.shape[1] != 224 or img.shape[2] !=3:
					print(img.shape)
				if '_h_' in fileName:
					X_person_dig.append(img)
				elif '_I_' in fileName:
					X_person_imp.append(None)
				else:
					X_person_normal.append(img)
			except Exception as ex:
				print(ex)
		if X_person_dig and X_person_imp and X_person_imp:
			X_dig.append(np.stack(X_person_dig))
			X_plain.append(np.stack(X_person_normal))
	assert(len(X_plain) == len(X_dig)) # 1 validation & 1 training sample per person
	return (X_plain, X_dig)


def getNormalGenerator(X_imposter, batch_size, infinite=True):
	while True:
		X_left, X_right, Y = [], [], []
		for i in range(len(X_imposter)):
			for j in range(len(X_imposter)):
				for x in X_imposter[i]:
					for y in X_imposter[j]:
						X_left.append(x)
						X_right.append(y)
						if i == j:
							Y.append([1])
						else:
							Y.append([0])
						if len(Y) == batch_size:
							yield [np.stack(X_left), np.stack(X_right)], np.stack(Y)
							X_left, X_right, Y = [], [], []
		if not infinite:
			break


def getImposterGenerator(X_plain, X_imposter, batch_size, infinite=True):
	while True:
		X_left, X_right, Y = [], [], []
		for person in X_plain:
			for x in person:
				for imposter in X_imposter:
					for y in imposter:
						X_left.append(x)
						X_right.append(y)
						Y.append([0])
						if len(Y) == batch_size:
							yield [np.stack(X_left), np.stack(X_right)], np.stack(Y)
							X_left, X_right, Y = [], [], []
		if not infinite:
			break


def getGenerator(norGen, normImpGen, impGen, batch_size, type=0, val_ratio=0.2):
	X_left, X_right, Y_send = [], [], []
	while True:
		X1, Y1 = norGen.next()
		X2, Y2 = normImpGen.next()
		X3, Y3 = impGen.next()
		Y = np.concatenate((Y1, Y2, Y2), axis=0)
		X = [np.concatenate((X1[0], X2[0], X3[0]), axis=0), np.concatenate((X1[1], X2[1], X3[1]), axis=0)]
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
		if len(Y_send) > 0:
			X_left = np.concatenate((X_left, X[0]), axis=0)
			X_right = np.concatenate((X_right, X[1]), axis=0)
			Y_send = np.concatenate((Y_send, Y), axis=0)
		else:
			X_left = np.copy(X[0])
			X_right = np.copy(X[1])
			Y_send = np.copy(Y)
		if len(Y_send) >= batch_size:
			yield ([X_left, X_right], Y_send)
			X_left, X_right, Y_send = [], [], []


def splitDisguiseData(X_dig, pre_ratio=0.5):
	X_dig_pre, X_dig_post = [], []
	for i in range(len(X_dig)):
		splitPoint = int(X_dig[i].shape[0] * pre_ratio)
		indices = np.arange(X_dig[i].shape[0])
		X_dig_pre.append(X_dig[i][indices[:splitPoint]])
		X_dig_post.append(X_dig[i][indices[splitPoint:]])
	return (X_dig_pre, X_dig_post)


def createMiniBatch(X_plain, X_dig):
	X_left, X_right, Y = [], [], []
	for i in range(len(X_plain)):
		for j in range(len(X_dig)):
			for x in X_plain[i]:
				for y in X_dig[j]:
					X_left.append(x)
					X_right.append(y)
					if i == j:
						Y.append([1])
					else:
						Y.append([0])
	for i in range(len(X_dig)):
		for j in range(len(X_dig)):
			for x in X_dig[i]:
				for y in X_dig[j]:
					X_left.append(x)
					X_right.append(y)
					if i == j:
						Y.append([1])
					else:
						Y.append([0])
	return [np.stack(X_left), np.stack(X_right)], np.stack(Y)


def getAllTestdata(prefix, fileList):
	allImages = os.path.join(prefix, trainFolder)
	X_plain = []
	X_dig = []
	personList = sorted(os.listdir(allImages))
	for person in personList:
		X_person_dig = []
		X_person_imp = []
		X_person_normal = []
		dirPath = os.path.join(allImages, person)
		faceList = sorted(os.listdir(dirPath))
		for impath in faceList:
			fullName = os.path.join(dirPath, impath)
			fullName = re.sub(r"[/]\s", "/", fullName)
			fileName = impath.rsplit('.', 1)[0]
			try:
				img = cv2.resize(np.asarray(Image.open(lookupFile(fullName)).convert('RGB'), dtype=np.float32), imageRes)
				if img.shape[0] !=224 or img.shape[1] != 224 or img.shape[2] !=3:
					print(img.shape)
				if '_h_' in fileName:
					X_person_dig.append(img)
				elif '_I_' in fileName:
					X_person_imp.append(None)
				else:
					X_person_normal.append(img)
			except Exception as ex:
				print(ex)
		if X_person_dig and X_person_imp and X_person_imp:
			X_dig.append(np.stack(X_person_dig))
			X_plain.append(np.stack(X_person_normal))
	assert(len(X_plain) == len(X_dig)) # 1 validation & 1 training sample per person
	return (X_plain, X_dig)

if __name__ == "__main__":
	import sys
	prefix = sys.argv[1]
	trainBoxesPath = prefix + "Training_data_face_coordinate.txt"
	boxMap = constructIndexMap(trainBoxesPath)
	#cropAllFolders(prefix, "Training_data", boxMap)
	#getAllTrainData(prefix, "Training_data")
