import os
import sys
import numpy as np

# Set seed for reproducability
np.random.seed(42)

def create_two_directories(baseDir, imagesDir, fileList, ratio=0.8):
    trainPath = os.path.join(baseDir, "train")
    valPath = os.path.join(baseDir, "val")
    os.mkdir(trainPath, 0755)
    os.mkdir(valPath, 0755)
    paths = []
    with open(fileList, 'r') as f:
        for line in f:
            paths.append(line.rstrip('\n'))

    postingList = {}
    for path in paths:
        personName = path.split('_')[0]
        postingList[personName] = postingList.get(personName, []) + [path]
    
    trainSplit = []
    valSplit = []
    for person in postingList.keys():
        trainSplit += postingList[person][: int(ratio * len(postingList[person]))]
        valSplit += postingList[person][int(ratio * len(postingList[person])):]
    for image in trainSplit:
        os.rename(os.path.join(imagesDir, image), os.path.join(trainPath, image))
    for image in valSplit:
        os.rename(os.path.join(imagesDir, image), os.path.join(valPath, image))


create_two_directories(sys.argv[1], sys.argv[2], sys.argv[3])
