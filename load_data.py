import numpy as np
import Image
import os
import cv2
from tqdm import tqdm

# Set seed for reproducability
np.random.seed(42)


def resize(images, new_size):
	resized_images = []
	images_tr *= 255.0
	images_tr = images_tr.astype("int32") 
	for image in images:
		resized_images.append(cv2.resize(image, new_size))
	resized_images = np.array(resized_images)
	resized_images = resized_images.astype("float32")
	resized_images /= 255.0
	return np.array(resized_images)


def preprocess_image(X):
	return X.astype("float32") / 255.0


def construct_data(images_dir):
	Y = []
	X = []
	for image_path in tqdm(os.listdir(images_dir)):
		person_class =  image_path.split('.')[0].split('_')[0]
		img = Image.open(os.path.join(images_dir, image_path))
		img.load()
		image = np.asarray( img, dtype="int32" )
		X.append(preprocess_image(image))
		Y.append(int(person_class))
	X = np.array(X)
	Y = np.array(Y)
	return X, Y		


def data_split(X, Y, pool_split=0.8):
	nb_classes = Y.shape[1]
	distr = {}
	for i in range(nb_classes):
		distr[i] = []
	if Y.shape[1] == nb_classes:
		for i in range(len(Y)):
			distr[np.argmax(Y[i])].append(i)
	else:
		for i in range(len(Y)):
			distr[Y[i][0]].append(i)
	X_bm_ret = []
	Y_bm_ret = []
	X_pm_ret = []
	Y_pm_ret = []
	# Calculate minimum number of points per class
	n_points = min([len(distr[i]) for i in distr.keys()])
	for key in distr.keys():
		st = np.random.choice(distr[key], n_points, replace=False)
		bm = st[:int(len(st)*pool_split)]
		pm = st[int(len(st)*pool_split):]
		X_bm_ret.append(X[bm])
		Y_bm_ret.append(Y[bm])
		X_pm_ret.append(X[pm])
		Y_pm_ret.append(Y[pm])
	X_bm_ret = np.concatenate(X_bm_ret)
	Y_bm_ret = np.concatenate(Y_bm_ret)
	X_pm_ret = np.concatenate(X_pm_ret)
	Y_pm_ret = np.concatenate(Y_pm_ret)
	return X_bm_ret, Y_bm_ret, X_pm_ret, Y_pm_ret
	

def split_into_sets(X, Y, lowres=(32,32), pool_ratio=0.75, big_ratio=0.5):
	X_unlabelled, Y_unlabelled, rest_x, rest_y = data_split(X, Y, pool_ratio)
	X_hr, Y_hr, X_lr, Y_lr = data_split(rest_x, rest_y, big_ratio)
	# Downsize images for LR format
	X_lr = resize(X_lr, lowres)
	return (X_unlabelled, Y_unlabelled), (X_hr, Y_hr), (X_lr, Y_lr)


if __name__ == "__main__":
	import sys
	X, Y = construct_data(sys.argv[1])
	