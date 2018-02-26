import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

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
	scores = [0.0 for _ in range(N)]
	count = 0
	while True:
		try:
			X_low, X_high, Y = testGenerator.next()
		except:
			break
		if resType == "low":
			Y_pred = model.predict(X_low)
		else:
			Y_pred = model.predict(X_high)
		count += len(Y_pred)
		for i in range(len(Y_pred)):
			sorted_preds = np.argsort(-Y_pred[i])
			for j in range(N):
				if mapping[Y_labels[i]] == sorted_preds[j]:
					score += 1.0
	return score / count
