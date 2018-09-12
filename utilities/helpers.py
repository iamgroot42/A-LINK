import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from keras_vggface import utils
from imgaug import augmenters as iaa


# Plot confusion matrix
def plot_confusion_matrix(cm, classes,title='Confusion matrix',cmap=plt.cm.Blues):
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				horizontalalignment="center",
				color="white" if cm[i, j] > thresh else "black")
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

# Convert numeric-labels to one-hot representation
def one_hot(Y, n_classes):
	y_ = np.zeros((len(Y), n_classes))
	y_[np.arange(len(Y)), Y] = 1
	return y_

# Change from [0,1] value to 0/1 value:
def roundoff(Y):
	y_ = []
	for y in Y:
		if y >= 0.5:
			y_.append([1])
		else:
			y_.append([0])
	return np.stack(y_)

# Convert from classnames into one-hot
def names_to_onehot(Y, mapping):
	Y_ = [ mapping[x] for x in Y]
	return one_hot(Y_, len(mapping))

# Preprocess data for passing to high-resolution model
def hr_preprocess(X):
	X_temp = np.copy(X)
	return utils.preprocess_input(X_temp, version=1)

# Map predictions in one class-labelling to another
def get_transformed_predictions(Y_true, mapping):
	ret = []
	for y in Y_true:
		ret.append(mapping[y])
	return np.array(ret)

# Calculate accuracy, given mappings between two class nomenclatures
def calculate_accuracy(Y_pred, Y_labels, mapping):
	score = 0.0
	for i in range(len(Y_pred)):
		if Y_pred[i] == mapping[Y_labels[i]]:
			score += 1.0
	return score / len(Y_pred)

# Calculate top-N accuracy
def calculate_topNaccuracy(Y_pred, Y_labels, mapping, N=5):
	score = 0.0
	for i in range(len(Y_pred)):
		sorted_preds = np.argsort(-Y_pred[i])
		for j in range(N):
			if mapping[Y_labels[i]] == sorted_preds[j]:
				score += 1.0
	return score / len(Y_pred)

# Calculate top N accuracy
def calculate_accuracy(testGenerator, model, resType, mapping, N=5):
	score = 0.0
	count = 0
	while True:
		try:
			X_low, X_high, Y = testGenerator.next()
		except Exception, e:
			print e
			break
		if resType == "low":
			Y_pred = model.predict(X_low)
		else:
			Y_pred = model.predict(X_high)
		count += len(Y_pred)
		for i in range(len(Y_pred)):
			sorted_preds = np.argsort(-Y_pred[i])
			for j in range(N):
				if mapping[Y[i]] == sorted_preds[j]:
					score += 1.0
	return score / count

# Shuffle data and split (in unison) into two parts
def unisonSplit(X, Y, leftRatio=0.4):
	indices = np.random.permutation(len(X))
	leftThreshold = int(len(X) * leftRatio)
	X_left, Y_left = X[indices[:leftThreshold]], Y[indices[:leftThreshold]]
	X_right, Y_right = X[indices[leftThreshold:]], Y[indices[leftThreshold:]]
	return (X_left, Y_left), (X_right, Y_right)

# Augment images with same transformations together
def augment_data(dataset, dataset_labels, augementation_factor=1, use_random_rotation=True, use_random_shear=True, use_random_shift=True):
	augmented_image_left = []
	augmented_image_right = []
	augmented_image_labels = []

	for num in range (0, dataset[0].shape[0]):
		for i in range(0, augementation_factor):
			# original image:
			augmented_image_left.append(dataset[0][num])
			augmented_image_right.append(dataset[1][num])
			augmented_image_labels.append(dataset_labels[num])

			if use_random_rotation:
				augmented_image_left.append(tf.contrib.keras.preprocessing.image.random_rotation(dataset[0][num], 20, row_axis=0, col_axis=1, channel_axis=2))
				augmented_image_right.append(tf.contrib.keras.preprocessing.image.random_rotation(dataset[1][num], 20, row_axis=0, col_axis=1, channel_axis=2))
				augmented_image_labels.append(dataset_labels[num])

			if use_random_shear:
				augmented_image_left.append(tf.contrib.keras.preprocessing.image.random_shear(dataset[0][num], 0.2, row_axis=0, col_axis=1, channel_axis=2))
				augmented_image_right.append(tf.contrib.keras.preprocessing.image.random_shear(dataset[1][num], 0.2, row_axis=0, col_axis=1, channel_axis=2))
				augmented_image_labels.append(dataset_labels[num])

			if use_random_shift:
				augmented_image_left.append(tf.contrib.keras.preprocessing.image.random_shift(dataset[0][num], 0.2, 0.2, row_axis=0, col_axis=1, channel_axis=2))
				augmented_image_right.append(tf.contrib.keras.preprocessing.image.random_shift(dataset[1][num], 0.2, 0.2, row_axis=0, col_axis=1, channel_axis=2))
				augmented_image_labels.append(dataset_labels[num])

	return [np.array(augmented_image_left), np.array(augmented_image_right)], np.array(augmented_image_labels)
