# ROC genenration script : provided by DFW
from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')

import  os
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

#Score matrix of dimension 7771 x 7771 obtained on test set containing similarity scores following the same format as the masked matrix
#Replace the name with the name of your score file.
score_matrix = np.loadtxt('TestScores.out', dtype=float) # matrix containing similarity scores
score_matrix = 1. - score_matrix

#Testing_data_combine_mask_matrix.txt is the provided mask matrix. Give path of the mask matrix.  
masked_matrix = np.loadtxt('testingMaskMatrix.txt', dtype=int) # load the mask matrix

#Threshold_weight.txt is provided - it contains the threshold values. Give path of the threshold file.
Threshold_weight = np.loadtxt('thresholds.txt', dtype=float)
#------------------------------------------------------------------------------------------------------------

#case 3 Generating Genuine and Imposter score for Overall Accuracy
Genuine_score= []
Imposter_score= []
for i in range(7771):
    z=i+1
    for j in range(z,7771):        
        if masked_matrix[i][j] == 1:
            Genuine_score.append(score_matrix[i][j])
        elif masked_matrix[i][j] == 2:
            Imposter_score.append(score_matrix[i][j])

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
print ('True_positive_rate and False_positive_rate generated')

# plotting 1000 point ROC  
plt.plot(false_positive_rate,true_positive_rate)
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title("ROC Curve", fontsize=14)
plt.xscale('log')
fig1 = plt.gcf()
#plt.show() 
#plt.draw()
fig1.savefig("1000_point_ROC_case1",dpi=100) # change the name by which to save figure accordingly to your requirement

