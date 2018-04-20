from __future__ import division
from __future__ import print_function

import  os
import sys
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

score_matrix = np.loadtxt(sys.argv[1], dtype=float) # matrix containing similarity scores

masked_matrix = np.loadtxt('updated_testing_mask.txt', dtype=int) # load the mask matrix

#Threshold_weight.txt is provided - it contains the threshold values. Give path of the threshold file.
Threshold_weight = np.loadtxt('thresholds.txt', dtype=float)
roc_case = int(sys.argv[3])

Genuine_score= []
Imposter_score= []
for i in range(7771):
	z=i+1
	for j in range(z,7771):
		if roc_case == 3:
			if masked_matrix[i][j] == 1 or masked_matrix[i][j] == 2:
				Genuine_score.append(score_matrix[i][j])
			elif masked_matrix[i][j] == 3 or masked_matrix[i][j] == 4:
				Imposter_score.append(score_matrix[i][j])
		elif roc_case == 2:
			if masked_matrix[i][j] == 2:
				Genuine_score.append(score_matrix[i][j])
			elif masked_matrix[i][j] == 4:
				Imposter_score.append(score_matrix[i][j])
		elif roc_case == 1:
			if masked_matrix[i][j] == 1:
				Genuine_score.append(score_matrix[i][j])
			elif masked_matrix[i][j] == 3:
				Imposter_score.append(score_matrix[i][j])
		else:
			print("Son, you screwed up.")
			exit()

print ('Genuine and Imposter score generated')
#------------------------------------------------------------------------------------------------------------

false_positive_rate=[]
true_positive_rate=[]

#Loop to generate true_positive_rate and false_positive_rate for different threshold values
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

# Save TPR, FPR
np.savetxt(sys.argv[2], np.array([true_positive_rate, false_positive_rate]))

