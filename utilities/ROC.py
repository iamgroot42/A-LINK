from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')

import sys
import numpy as np
import matplotlib.pyplot as plt

for input in sys.argv[1:-1]:
	TPR, FPR = np.loadtxt(input)
	plt.plot(FPR, TPR, label=input)

plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title("ROC Curve", fontsize=14)
plt.legend()
plt.savefig(sys.argv[-1], dpi=500)
