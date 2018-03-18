import siamese
from tqdm import tqdm
import numpy as np


if __name__ == "__main__":
	disguisedFacesModel = siamese.SiameseNetwork((2048,), "disguisedModel", 0.1)
	disguisedFacesModel.maybeLoadFromMemory()
	features = np.load("processeData.npy")
	scores = []
	assert(features.shape[0] == 7771)
	for i in tqdm(range(len(features))):
		X_left, X_right = [], []
		for x in features:
			X_left.append(features[i])
			X_right.append(x)
		numbers = [ out[0] for out in disguisedFacesModel.predict([np.stack(X_left), np.stack(X_right)])]
		scores.append(numbers)
	scores = np.stack(scores)
	np.savetxt('TestScores.out', scores)

