import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import cv2
import siamese
import re

# Don't hog GPU
import tensorflow as tf
import keras
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
keras.backend.set_session(sess)


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
		print fullPath
		print os.listdir(directory)
		return None


def generatePredictions(prefix, filePaths, featureModel, imageRes=(224, 224)):
	features = []
	for file in tqdm(filePaths):
		fullName = os.path.join(prefix, file) #.decode('utf8')
		fullName = re.sub(r"[/]\s", "/", fullName)
		try:
			img = cv2.resize(np.asarray(Image.open(lookupFile(fullName)).convert('RGB'), dtype=np.float32), imageRes)
			if img.shape[0] !=224 or img.shape[1] != 224 or img.shape[2] !=3:
                                print img.shape
			features.append(featureModel.process([img])[0])
		except Exception, e:
			print e
	return np.stack(features)


if __name__ == "__main__":
	import sys
	prefix = sys.argv[1]
	pathFile = os.path.join(prefix, "Testing_data_face_name.txt")
	with open(pathFile) as f:
		names = f.readlines()
	names = [r.rstrip() for r in names]
	featureModel = siamese.RESNET50((224, 224))
	features = generatePredictions(prefix, names, featureModel)
	np.save("processedData.npy", features)
