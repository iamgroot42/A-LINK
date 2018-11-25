import os
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm


def readAllImages(dirPath):
	person_wise = {}
	for path in tqdm(os.listdir(dirPath)):
		if "_051_06" in path or "_051_08" in path:
			img = cv2.resize(np.asarray(Image.open(os.path.join(dirPath, path)).convert('RGB'), dtype=np.float32), (150, 150))
			person_id = int(path.split('_')[0])
			person_wise[person_id] = person_wise.get(person_id, []) + [img]
	return person_wise


def personsplit(person_wise, split_ratio=0.25, target_resolution=(48, 48), num_train=100):
	shuffled_indices = np.random.choice(person_wise.keys(), len(person_wise.keys()), replace=False)
	train_indices =shuffled_indices[:num_train]
	test_indices  = shuffled_indices[num_train:]

	# Make splits for labelled/unlabelled data
	split_pre, split_post = {}, {}
	for i in range(len(train_indices)):
		splitPoint = int(len(person_wise[train_indices[i]]) * split_ratio)
		indices = np.arange(len(person_wise[train_indices[i]]))
		if train_indices[i] in split_pre:
			split_pre[train_indices[i]]  = np.concatenate((split_pre[train_indices[i]], np.array(person_wise[train_indices[i]])[indices[:splitPoint]]))
			split_post[train_indices[i]] = np.concatenate((split_post[train_indices[i]],np.array(person_wise[train_indices[i]])[indices[splitPoint:]]))
		else:
			split_pre[train_indices[i]]  = np.array(person_wise[train_indices[i]])[indices[:splitPoint]]
			split_post[train_indices[i]] = np.array(person_wise[train_indices[i]])[indices[splitPoint:]]

	# Contruct siamese-compatible data
	def mix_match_data(data, base, resize=False):
		data_X_left, data_X_right, data_Y = [], [], []
		for i in tqdm(range(len(data))):
			# All images of that person
			for i_sub in base[data[i]]:
				if resize:
					i_sub_smaller = cv2.resize(i_sub, target_resolution)
				else:
					i_sub_smaller = i_sub
				for j in range(i, len(data)):
					# All images of that person
					for j_sub in base[data[j]]:
						if resize:
							j_sub_smaller = cv2.resize(j_sub, target_resolution)
						else:
							j_sub_smaller = j_sub
						data_X_left.append(i_sub_smaller)
						data_X_right.append(j_sub_smaller)
						if i == j:
							data_Y.append([1])
						else:
							data_Y.append([0])
		return (data_X_left, data_X_right), data_Y

	train_X, train_Y = mix_match_data(train_indices, split_pre)
	pool_X, pool_Y   = mix_match_data(train_indices, split_post)
	# TODO : Use only the test data that they used (for consistent comparison)
	test_X, test_Y   = mix_match_data(test_indices, person_wise, resize=True)

	# Free up RAM
	del person_wise

	return (train_X, train_Y), (pool_X, pool_Y), (test_X, test_Y)


def generatorFeaturized(X, Y, batch_size, featurize=None, resize_res=None):
	while True:
		for i in range(0, len(X), batch_size):
			X_left  = X[0][i: i + batch_size]
			X_right = X[1][i: i + batch_size]
			Y       = Y[i: i + batch_size]
			if resize_res:
				X_left, X_right = resizeImages([X_left, X_right], resize_res)
			if featurize:
				X_left = featurize.process(X_left)
				X_right = featurize.process(X_right)
			yield [X_left, X_right], Y


def resizeImages(images, resize_res):
	resized_left  = [cv2.resize(image, resize_res) for image in images[0]]
	resized_right = [cv2.resize(image, resize_res) for image in images[1]]
	return [np.array(resized_left), np.array(resized_right)]
