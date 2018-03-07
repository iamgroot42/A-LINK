import numpy as np
from PIL import Image
import os
import cv2
import numpy as np


def cropImages(prefix, dirPath, faceBoxes):
	imagesBefore = os.listdir(os.path.join(prefix, dirPath))
	for imPath in imagesBefore:
		try:
			partialName = os.path.join(dirPath, imPath)
			fullName = os.path.join(prefix, partialName)
			img = Image.open(fullName)
			tx, h, w, by = faceBoxes[partialName]
			img = img.crop((tx, h, w, by))
			img.save(fullName)
		except:
			os.remove(fullName)
			print("File I/O error")


def constructIndexMap(filePath):
	with open(filePath, 'r') as f:
		mapping = {}
		for row in f:
			imgname, tx, h, w, by = row.rstrip('\n').rstrip().rsplit(' ', 4)
			mapping[imgname] = [float(x) for x in [tx, h, w, by]]
	return mapping


def cropAllFolders(prefix, trainFolder, boxMap):
	allImages = os.path.join(prefix, trainFolder)
	for person in os.listdir(allImages):
		cropImages(prefix, os.path.join(trainFolder, person), boxMap)


def getAllTrainData(prefix, trainFolder, imageRes, imposter=False):
	allImages = os.path.join(prefix, trainFolder)
	X_plain, Y_plain = [], []
	X_val, Y_val = [], []
	X_dig, Y_dig = [], []
	X_imp, Y_imp = [], []
	names = os.listdir(allImages)
	for person in os.listdir(allImages):
		dirPath = os.path.join(allImages, person)
		for impath in os.listdir(dirPath):
			fullName = os.path.join(dirPath, impath)
			fileName = impath.rsplit('.', 1)[0]
			try:
				img = cv2.resize(np.asarray(Image.open(fullName).convert('RGB'), dtype=np.float32), imageRes)
				if img.shape[0] !=224 or img.shape[1] != 224 or img.shape[2] !=3:
					print img.shape
				if '_h_' in fileName:
					X_dig.append(img)
					Y_dig.append(person)
				elif '_I_' in fileName:
					if imposter:
						X_imp.append(img)
						Y_imp.append(person)
				elif fileName[-2:] == '_a':
					X_val.append(img)
					Y_val.append(person)
				else:
					X_plain.append(img)
					Y_plain.append(person)
			except Exception, e:
				print(e)
	if imposter:
		return names, (np.stack(X_plain), np.stack(Y_plain)), (np.stack(X_val), np.stack(Y_val)), (np.stack(X_dig), np.stack(Y_dig)), (np.stack(X_imp), np.stack(Y_imp))
	else:
		return names, (np.stack(X_plain), np.stack(Y_plain)), (np.stack(X_val), np.stack(Y_val)), (np.stack(X_dig), np.stack(Y_dig))


if __name__ == "__main__":
	prefix = "DFW/DFW_Data/"
	trainBoxesPath = prefix + "Testing_data_face_coordinate.txt"
	boxMap = constructIndexMap(trainBoxesPath)
	#cropAllFolders(prefix, "Testing_data", boxMap)
	getAllTrainData(prefix, "Training_data")
