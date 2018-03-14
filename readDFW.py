import numpy as np
from PIL import Image
import os
import cv2
import numpy as np


def cropImages(prefix, dirPath, faceBoxes):
	imagesBefore = os.listdir(os.path.join(prefix, dirPath))
	i = 0
	for imPath in imagesBefore:
		try:
			partialName = os.path.join(dirPath, imPath)
			fullName = os.path.join(prefix, partialName)
			img = Image.open(fullName).convert('RGB')
			tx, h, w, by = faceBoxes[partialName]
			img = img.crop((tx, h, w, by))
			#img.save(fullName)
		except:
			#os.remove(fullName)
			i += 1
			#print("File I/O error")
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
	for person in os.listdir(allImages):
		prob += cropImages(prefix, os.path.join(trainFolder, person), boxMap)
	print("Problem with", prob)


def getAllTrainData(prefix, trainFolder, imageRes):
	allImages = os.path.join(prefix, trainFolder)
	X_plain = []
	X_dig = []
	X_imp = []
	for person in os.listdir(allImages):
		X_person_dig = []
		X_person_imp = []
		X_person_normal = []
		dirPath = os.path.join(allImages, person)
		for impath in os.listdir(dirPath):
			fullName = os.path.join(dirPath, impath)
			fileName = impath.rsplit('.', 1)[0]
			try:
				img = cv2.resize(np.asarray(Image.open(fullName).convert('RGB'), dtype=np.float32), imageRes)
				if img.shape[0] !=224 or img.shape[1] != 224 or img.shape[2] !=3:
					print img.shape
				if '_h_' in fileName:
					X_person_dig.append(img)
				elif '_I_' in fileName:
					X_person_imp.append(img)
				else:
					X_person_normal.append(img)
			except Exception, e:
				print(e)
		X_dig.append(np.stack(X_person_dig))
		X_imp.append(np.stack(X_person_imp))
		X_plain.append(np.stack(X_person_normal))
	assert(len(X_plain) == len(X_val)) # 1 validation & 1 training sample per person
	X_plain = np.stack(X_plain)
	X_val = np.stack(X_val)
	return (X_plain, X_dig, X_imp)


def getAllTestData(prefix, trainFolder, imageRes):
	allImages = os.path.join(prefix, trainFolder)
	X = []
	Y = []
	for person_index, person in enumerate(os.listdir(allImages)):
		dirPath = os.path.join(allImages, person)
		for impath in os.listdir(dirPath):
			fullName = os.path.join(dirPath, impath)
			fileName = impath.rsplit('.', 1)[0]
			try:
				img = cv2.resize(np.asarray(Image.open(fullName).convert('RGB'), dtype=np.float32), imageRes)
				if img.shape[0] !=224 or img.shape[1] != 224 or img.shape[2] !=3:
					print img.shape
				X.append(img)
				if '_h_' in fileName:
					Y.append(person_index)
				elif '_I_' in fileName:
					Y.append(-person_index)
				else:
					Y.append(person_index)
			except Exception, e:
				print(e)
	X = np.stack(X)
	Y = np.stack(Y)
	return (X, Y)

def getNormalGenerator(X_imposter, batch_size):
	while True:
		X_left, X_right, Y = [], [], []
		for i in range(len(X_imposter)):
			for j in range(len(X_imposter)):
				for x in X_imposter[i]:
					for y in X_imposter[j]:						
						X_left.append(x)
						X_right.append(y)
						if i == j:
							Y.append([0 1])
						else:
							Y.append([1 0])
				if len(Y) == batch_size:
					yield [np.stack(X_left), np.stack(X_right)], np.stack(Y)
					X_left, X_right, Y = [], [], []


def getImposterGenerator(X_plain, X_imposter, batch_size):
	while True:
		X_left, X_right, Y = [], [], []
		for x in X_plain:
			for imposter in X_imposter:
				for y in imposter:
				X_left.append(x)
				X_right.append(y)
				Y.append([1 0])
				if len(Y) == batch_size:
					yield [np.stack(X_left), np.stack(X_right)], np.stack(Y)
					X_left, X_right, Y = [], [], []


def getGenerator(X_plain, X_imposter, batch_size):
	norGen = getNormalGenerator(X_plain, batch_size)
	normImpGen = getNormalGenerator(X_imposter, batch_size)
	impGen  = getImposterGenerator(X_plain, X_imposter, batch_size)
	while True:
		X1, Y1 = norGen.next()
		X2, Y2 = normImpGen.next()
		X3, Y3 = impGen.next()
		Y = np.concatenate((Y1, Y2, Y2), axis=0)
		X = [np.concatenate((X1[0], X2[0], X3[0]), axis=0), np.concatenate((X1[1], X2[1], X3[1]), axis=0)
		yield X, Y


if __name__ == "__main__":
	prefix = "DFW/DFW_Data/"
	trainBoxesPath = prefix + "Training_data_face_coordinate.txt"
	boxMap = constructIndexMap(trainBoxesPath)
	cropAllFolders(prefix, "Training_data", boxMap)
	#getAllTrainData(prefix, "Training_data")
