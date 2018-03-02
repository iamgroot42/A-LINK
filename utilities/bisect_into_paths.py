import os
import sys
import numpy as np

# Set seed for reproducability
np.random.seed(42)

imagesDir = sys.argv[1]
filesDir = sys.argv[2]

UNLABELLED_RATIO = 0.2
TEST_RATIO = 0.6
HR_RATIO = 0.15
LR_RATIO = 0.05

# Check that all data is used up
assert(UNLABELLED_RATIO + TEST_RATIO + HR_RATIO + LR_RATIO == 1.0)

udPaths = []
tdPaths = []
hrPaths = []
lrPaths = []

for classFolder in os.listdir(imagesDir):
    classLabel = os.path.join(imagesDir, classFolder)
    imagePaths = os.listdir(classLabel)[:]
    np.random.shuffle(imagePaths)
    threshold1 = int(UNLABELLED_RATIO * len(imagePaths))
    threshold2 = int(TEST_RATIO * len(imagePaths)) + threshold1
    threshold3 = int(HR_RATIO * len(imagePaths)) + threshold2
    udPaths += imagePaths[:threshold1]
    tdPaths += imagePaths[threshold1:threshold2]
    hrPaths += imagePaths[threshold2:threshold3]
    lrPaths += imagePaths[threshold3:]
    #print len(imagePaths[:threshold1]), "for unlabelled"
    #print len(imagePaths[threshold1:threshold2]), "for test"
    #print len(imagePaths[threshold2:threshold3]), "for high res"
    #print len(imagePaths[threshold3:]), "for low res"

with open(os.path.join(filesDir, "unlabelledData.txt"), 'w') as f:
    for path in udPaths:
        f.write(path + '\n')

with open(os.path.join(filesDir, "testData.txt"), 'w') as f:
    for path in tdPaths:
        f.write(path + '\n')

with open(os.path.join(filesDir, "highResData.txt"), 'w') as f:
    for path in hrPaths:
        f.write(path + '\n')

with open(os.path.join(filesDir, "lowResData.txt"), 'w') as f:
    for path in lrPaths:
        f.write(path + '\n')
