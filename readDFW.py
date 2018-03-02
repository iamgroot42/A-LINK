import numpy as np
from PIL import Image
import os


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


def readAllFolders(prefix, trainFolder, boxMap):
	allImages = os.path.join(prefix, trainFolder)
	for person in os.listdir(allImages):
		print allImages, person
		cropImages(prefix, os.path.join(trainFolder, person), boxMap)


if __name__ == "__main__":
	prefix = "DFW/DFW_Data/"
	trainBoxesPath = prefix + "Testing_data_face_coordinate.txt"
	boxMap = constructIndexMap(trainBoxesPath)
	readAllFolders(prefix, "Testing_data", boxMap)
