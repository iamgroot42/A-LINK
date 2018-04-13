import load_data
import numpy as np
import siamese
import sys

import matplotlib
matplotlib.use('Agg')

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

# Don't hog GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
keras.backend.set_session(sess)

if __name__ == "__main__":
	type = int(sys.argv[2])
	imagesDir = 'data/'
	testDataList = 'fileLists/testData.txt'
	# Low-resolution model
	if type == 0:
		imageRes = (32, 32)
		lowResM = siamese.SmallRes(imageRes + (3,), (2048,), sys.argv[1], 0.1)
		lowResM.maybeLoadFromMemory()
		testGen = load_data.testDataGenerator(imagesDir, testDataList, imageRes, 64)
	# High-resolution model
	else:
		imageRes = (224, 224)
		highResM = siamese.SiameseNetwork((2048,), sys.argv[1], 0.1)
		conversionModel = siamese.RESNET50(imageRes)
		testGen = load_data.testDataGenerator(imagesDir, testDataList, imageRes, 64)
	actual = []
	predictions = []
	while True:
		try:
			X, Y = testGen.next()
		except:
			break
		if type != 0:
			X = conversionModel.process(X)
		X_siam, Y_siam = load_data.labelToSiamese(X, Y)
		if type == 0:
			Y_pred = lowResM.predict(X_siam)
		else:
			Y_pred = highResM.predict(X_siam)
		for i in range(len(Y_pred)):
			actual.append(Y_siam[i][0])
			predictions.append(Y_pred[i][0])
	
	fpr, tpr, _ = roc_curve(actual, predictions)
	roc_auc = auc(fpr, tpr)	
	plt.figure()
	lw = 2
	plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	#plt.xscale('log')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('log-ROC curve for MultiPIE')
	plt.legend(loc="lower right")
	plt.savefig(sys.argv[3])
