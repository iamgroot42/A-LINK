import numpy as np
from PIL import Image
import os
from itertools import tee
import cv2
import re


def cropImages(prefix, dirPath, faceBoxes):
	imagesBefore = sorted(os.listdir(os.path.join(prefix, dirPath)))
	i = 0
	for imPath in imagesBefore:
		try:
			partialName = os.path.join(dirPath, imPath).decode('utf-8').replace('\xef\xbb\xbf', '')
			fullName = os.path.join(prefix, partialName).decode('utf8')
			fullName =  re.sub(r"[/]\s", "/", fullName).replace('\xef\xbb\xbf', '')
			img = Image.open(fullName).convert('RGB')
			tx, h, w, by = faceBoxes[partialName]
			img = img.crop((tx, h, w, by))
			img.save(fullName)
		except Exception, e:
			os.remove(fullName)
			i += 1
			print(fullName.decode('utf8'))
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


def getAllTrainData(prefix, trainFolder, imageRes, model):
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
			fullName = os.path.join(dirPath, impath).decode('utf-8')
			fullName = re.sub(r"[/]\s", "/", fullName).replace('\xef\xbb\xbf', '')
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
		if X_person_dig and X_person_imp and X_person_normal:
			X_dig.append(model.process(np.stack(X_person_dig)))
			X_imp.append(model.process(np.stack(X_person_imp)))
			X_plain.append(model.process(np.stack(X_person_normal)))
	assert(len(X_plain) == len(X_dig) and len(X_dig) == len(X_imp)) # 1 validation & 1 training sample per person
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
                        fullName = os.path.join(dirPath, impath).decode('utf-8')
			fullName = re.sub(r"[/]\s", "/", fullName).replace('\xef\xbb\xbf', '')
                        fileName = impath.rsplit('.', 1)[0]
                        try:
                                img = cv2.resize(np.asarray(Image.open(fullName).convert('RGB'), dtype=np.float32), imageRes)
                                if img.shape[0] !=224 or img.shape[1] != 224 or img.shape[2] !=3:
                                        print img.shape
                                if '_h_' in fileName:
                                        X_person_dig.append(img)
                                elif '_I_' in fileName:
					X_person_imp.append(None)
                                else:
                                        X_person_normal.append(img)
                        except Exception, e:
                                print(e)
                if X_person_dig and X_person_imp and X_person_imp:
                        X_dig.append(np.stack(X_person_dig))
                        X_plain.append(np.stack(X_person_normal))
        assert(len(X_plain) == len(X_dig)) # 1 validation & 1 training sample per person
        return (X_plain, X_dig)


def getAllTestData(prefix, trainFolder, imageRes, model):
	allImages = os.path.join(prefix, trainFolder)
	X = []
	Y = []
	personList = sorted(os.listdir(allImages))
	for person_index, person in enumerate(personList):
		dirPath = os.path.join(allImages, person)
		faceList = sorted(os.listdir(dirPath))
		for impath in faceList:
			fullName = os.path.join(dirPath, impath)
			fullName = re.sub(r"[/]\s", "/", fullName).replace('\xef\xbb\xbf', '')
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
	return (model.process(X), Y)


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
							Y.append([1])
						else:
							Y.append([0])
						if len(Y) == batch_size:
							yield [np.stack(X_left), np.stack(X_right)], np.stack(Y)
							X_left, X_right, Y = [], [], []


def getImposterGenerator(X_plain, X_imposter, batch_size):
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


def getGenerator(X_plain, X_imposter, batch_size, val_ratio=0.2):
	norGen = getNormalGenerator(X_plain, batch_size)
	normImpGen = getNormalGenerator(X_imposter, batch_size)
	impGen  = getImposterGenerator(X_plain, X_imposter, batch_size)
	while True:
		X1, Y1 = norGen.next()
		X2, Y2 = normImpGen.next()
		X3, Y3 = impGen.next()
		Y = np.concatenate((Y1, Y2, Y2), axis=0)
		X = [np.concatenate((X1[0], X2[0], X3[0]), axis=0), np.concatenate((X1[1], X2[1], X3[1]), axis=0)]
		indices = np.random.permutation(len(Y))
		splitPoint = int(len(Y) * val_ratio)
		X_train, Y_train = [ x[indices[splitPoint:]] for x in X], Y[indices[splitPoint:]]
		X_test, Y_test = [ x[indices[:splitPoint]] for x in X], Y[indices[:splitPoint]]
		yield (X_train, Y_train), (X_test, Y_test)


def splitGen(sourceGen, train):
	for tuple in sourceGen:
		if train:
			yield tuple[0]
		else:
			yield tuple[1]


def getTrainValGens(X_plain, X_imposter, batch_size, val_ratio=0.2):
	dataGen1, dataGen2 = tee(getGenerator(X_plain, X_imposter, batch_size, val_ratio))
	trainGen, valGen = splitGen(dataGen1, True), splitGen(dataGen1, False)
	return trainGen, valGen


def splitDisguiseData(X_dig, pre_ratio=0.5):
	X_dig_pre, X_dig_post = [], []
	for i in range(len(X_dig)):
		splitPoint = int(X_dig[i].shape[0] * pre_ratio)
		#indices = np.random.permutation(X_dig[i].shape[0])
		indices = np.arange(X_dig[i].shape[0])
		#print(X_dig[i].shape[0], X_dig[i][indices[:splitPoint]].shape[0], X_dig[i][indices[splitPoint:]].shape[0])
		assert(len(X_dig) >= 2), "You dead, nigga" #Required, for the way model is currenty built
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
						Y.append([0, 1])
					else:
						Y.append([1, 0])
	for i in range(len(X_dig)):
		for j in range(len(X_dig)):
			for x in X_dig[i]:
				for y in X_dig[j]:
					X_left.append(x)
					X_right.append(y)
					if i == j:
						Y.append([0, 1])
					else:
						Y.append([1, 0])
	return [np.stack(X_left), np.stack(X_right)], np.stack(Y)


if __name__ == "__main__":
	prefix = "DFW/DFW_Data/"
	trainBoxesPath = prefix + "Testing_data_face_coordinate.txt"
	boxMap = constructIndexMap(trainBoxesPath)
	cropAllFolders(prefix, "Testing_data", boxMap)
	#getAllTrainData(prefix, "Training_data")
