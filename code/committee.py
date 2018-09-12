import cv2
import numpy as np


class Bagging:
	def __init__(self, models, attacks):
		self.models = models
		#assert(len(attacks) == len(self.models))
		self.attacks = []
		for attack in attacks:
			self.attacks.append(attack)

	def predict(self, predict_on):
		predictions = []
		for model in self.models:
			predictions.append(model.predict(predict_on))
		predicted = np.sum(np.array(predictions),axis=0) / len(self.models)
		predicted = np.array(predicted)
		return predicted

	def resize(self, images, new_size):
		resized_images = []
		for image in images:
			resized_images.append(cv2.resize(image, new_size))
		return np.array(resized_images)

	def attackModel(self, images, target_size):
		attackPairs = []
		# Heuristic to combine these attack sample points
		if len(self.attacks) == 1:
			return self.resize(self.attacks[0].addNoise(images), target_size)
		else:
			perturbed_images = []
			for attack in self.attacks:
				perturbed_images.append(self.resize(attack.addNoise(images), target_size))
			return perturbed_images

