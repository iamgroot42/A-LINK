import numpy as np
from sklearn import metrics
import sys

def find_nearest(array,value):
	idx = (np.abs(array-value)).argmin()
	return idx

TPR, FPR = np.loadtxt(sys.argv[1])
FNR = 1 - TPR
eer = FPR[np.nanargmin(np.absolute(FNR - FPR))]

auc = metrics.auc(FPR, TPR)
print("AUC %f" % auc)
print("EER %f" % eer)

value = 0.010
index_1_percent=find_nearest(FPR, value)
GAR_FAR_value=TPR[index_1_percent]
print ('GAR is %f for %f FAR' % (GAR_FAR_value, value))

value = 0.0010
index_0_1_percent=find_nearest(FPR, value)
GAR_FAR_value=TPR[index_0_1_percent]
print ('GAR is %f for %f FAR' % (GAR_FAR_value, value))
