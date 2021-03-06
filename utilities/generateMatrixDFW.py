import siamese
from tqdm import tqdm
import numpy as np

import tensorflow as tf
import keras

# Don't hog GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
keras.backend.set_session(sess)


if __name__ == "__main__":
	import sys
	if len(sys.argv) < 2:
		print("python " + sys.argv[0] + " modelName outputFilePath")
	disguisedFacesModel = siamese.SiameseNetwork((2048,), sys.argv[1], 0.1)
	if disguisedFacesModel.maybeLoadFromMemory():
		print("Loaded model successfully!")
	else:
		print("Oops! Model not found")
		exit()
	features = np.load("processedData.npy")
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
	np.savetxt(sys.argv[2], scores)
