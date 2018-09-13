from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')

import sys
import numpy as np
import matplotlib.pyplot as plt
import os

for input_type in os.listdir(sys.argv[1]):
	try:
		TPR, FPR = np.loadtxt(os.path.join(sys.argv[1], input_type))
		fileName = input_type.split('/')[-1].rsplit('.', 1)[0]
		fileName = '$%s$' % fileName
		fileName = fileName.replace(" ", "\ ")
		plt.plot(FPR, TPR,  label=fileName)
	except:
		pass

# Plot y=x line for random-reference
#plt.plot([0,1], [1,0], 'r--')

# Define title of plot
type = int(sys.argv[2])
if type== 1:
	plt.title("ROC Curve : Impersonation", fontsize=14)
elif type == 2:
	plt.title("ROC Curve : Obfuscation", fontsize=14)
elif type == 3:
	plt.title("ROC Curve : Overall", fontsize=14)
else:
	plt.title("ROC Curve", fontsize=14)

plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.legend()
plt.xscale('log')
plt.savefig(sys.argv[-1], dpi=500)
