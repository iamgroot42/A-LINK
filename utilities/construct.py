import h5py
from PIL import Image
import numpy as np
import os
import cv2
import sys


images_dir = sys.argv[1]
image_paths = os.listdir(images_dir)    

for i in range(len(image_paths)/512):
	X = []
	Y = []
	f = h5py.File('dummy.h5','a')
	for j in range(512):
		image_path = image_paths[j + i*512]
		person_class = image_path.split('.')[0].split('_')[0]
		X.append(np.asarray(Image.open(os.path.join(images_dir, image_path)), dtype="int32"))
		Y.append(int(person_class)-1)
	X = np.array(X).astype('float32')
	Y = np.array(Y)
	f.create_dataset('multiPIE50k_batch_' + str(i) + '_X', data=X)
	f.create_dataset('multiPIE50k_batch_' + str(i) + '_Y', data=Y)
	f.close()
	print 'Covered',(i+1)*512,'images'

