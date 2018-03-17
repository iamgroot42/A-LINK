import numpy as np
from PIL import Image
import os
from itertools import tee
import cv2
import siamese
import re


def generatePredictions(prefix, filePaths, imageRes=(224, 224)):
	for file in filePaths:
		fullName = os.path.join(prefix, file).decode('utf8')
		fullName = re.sub(r"[/]\s", "/", fullName)
		try:
			img = cv2.resize(np.asarray(Image.open(fullName).convert('RGB'), dtype=np.float32), imageRes)
			if img.shape[0] !=224 or img.shape[1] != 224 or img.shape[2] !=3:
				print img.shape
		except Exception, e:
			print e


if __name__ == "__main__":
	import sys
	prefix = "DFW/DFW_Data"
	pathFile = os.path.join(prefix, "Testing_data_face_name.txt")
	with open(pathFile) as f:
		names = f.readlines()
	names = [r.rstrip() for r in names]
	generatePredictions(prefix, names)
