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


	#Threshold_weight.txt is provided - it contains the threshold values. Give path of the threshold file.
	Threshold_weight = np.loadtxt('thresholds.txt', dtype=float)

	Genuine_score= []
	Imposter_score= []
	for i in range(len(predictions)):
		if actual[i] == 1:
			Genuine_score.append(predictions[i])
		else:
			Imposter_score.append(predictions[j])
	print ('Genuine and Imposter score generated')			
	
	false_positive_rate=[]
	true_positive_rate=[]
	for i in range(len(Threshold_weight)):
		True_positive = 0
		False_positive = 0
		threshold_value = Threshold_weight[i]
		for z in range(len(Genuine_score)):
			 if Genuine_score[z] >= threshold_value:
				True_positive = True_positive+1
		for z1 in range(len(Imposter_score)):
			if Imposter_score[z1] >= threshold_value:
				False_positive=False_positive+1
		False_PR=False_positive/len(Imposter_score)
		false_positive_rate.append(False_PR)
		True_PR=True_positive/len(Genuine_score)
		true_positive_rate.append(True_PR)
	plt.plot(false_positive_rate, true_positive_rate)
	plt.xlabel('False Positive Rate', fontsize=14)
	plt.ylabel('True Positive Rate', fontsize=14)
	plt.title("ROC Curve", fontsize=14)
	#plt.legend()
	plt.savefig(sys.argv[3], dpi='200')
