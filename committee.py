import cv2
import numpy as np


class Bagging:
	def __init__(self, n_classes, models_dir, attacks):
		self.n_classes = n_classes
		assert(len(attacks) == len(os.listdir(self.models_dir)))
		self.models_dir = models_dir
		for file in os.listdir(self.models_dir):
			self.models.append(load_model(os.path.join(self.models_dir,file)))
		self.attacks = []
		for i, attack in enumerate(attacks):
			self.attacks.append(attack(self.models[i]))

	def predict(self, predict_on, method='voting'):
		predictions = []
		for model in self.models:
			predictions.append(model.predict(predict_on))
		if method == 'voting':
			ultimate = [ {i:0 for i in range(self.n_classes)} for j in range(len(predict_on))]
			for prediction in predictions:
				for i in range(len(prediction)):
					ultimate[i][np.argmax(prediction[i])] += 1
			predicted = []
			for u in ultimate:
				voted = sorted(u, key=u.get, reverse=True)
				predicted.append(voted[0])
			predicted = keras.utils.to_categorical(np.array(predicted), self.n_classes)
			return predicted
		else:
			predicted = np.argmax(np.sum(np.array(predictions),axis=0),axis=1)
			predicted = keras.utils.to_categorical(np.array(predicted), self.n_classes)
		return predicted

	def resize(self, images, new_size):
		resized_images = []
		for image in images:
			resized_images.append(cv2.resize(image, new_size))
		return np.array(resized_images)

	def attackModel(self, images, noiseFunc, target_size):
		noisy_images = noiseFunc.addNoise(images)
		return self.resize(noisy_images, target_size)

	def attackModel(self, images, target_size):
		attackPairs = []
		for i, model in enumerate(self.models):
			X = self.attacks[i].addNoise(images)
			Y = model.predict(images)
			attackPairs.append(X, Y)
		# Heuristic to combine these attack sample points
		finalizedImages = []
		return self.resize(finalizedImages, target_size)
