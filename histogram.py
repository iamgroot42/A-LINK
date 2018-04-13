import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import sys

score_matrix = np.loadtxt(sys.argv[1], dtype=float)
masked_matrix = np.loadtxt('testingMaskMatrix.txt', dtype=int)

newmat = score_matrix

Genuine_score= []
Imposter_score= []
for i in range(7771):
    z=i+1
    for j in range(z,7771):
        if masked_matrix[i][j] == 1:
            Genuine_score.append(newmat[i][j])
        elif masked_matrix[i][j] == 2:
            Imposter_score.append(newmat[i][j])
#plt.hist(newmat, normed=True)
#plt.savefig('histogram.png')

#bins = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
#hist, bin_edges = np.histogram(newmat, bins=bins)

print np.sort(Genuine_score), "genuine"
print np.sort(Imposter_score), "imposter"

plt.hist(Genuine_score, bins=20, range=(0.0, 1.0), label='Genuine', alpha=0.5)
plt.hist(Imposter_score, bins=20, range=(0.0, 1.0), label='Imposter', alpha=0.5)
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='upper right')
plt.savefig('Histogram.png')
