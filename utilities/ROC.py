from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')

import sys
import numpy as np
import matplotlib.pyplot as plt

for input in sys.argv[1:-1]:
	TPR, FPR = np.loadtxt(input)
	plt.plot(FPR, TPR,  label=input.split('/')[-1].rsplit('.', 1)[0])

# Plot y=x line for random-reference
plt.plot([0,1], [0,1], 'r--')
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title("ROC Curve", fontsize=14)
plt.legend()
#plt.xscale('log')
plt.savefig(sys.argv[-1], dpi=500)
