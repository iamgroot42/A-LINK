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
		# Prediction-averaging across ensemble of models
		predicted = np.sum(np.array(predictions), axis=0) / len(self.models)
		predicted = np.array(predicted)
		return predicted

	def resize(self, images, new_size):
		resized_images = []
		for image in images:
			resized_images.append(cv2.resize(image, new_size))
		return np.array(resized_images)

	def attackModel(self, image_pairs, target_size, target_labels=None):
		# Heuristic to combine these attack sample points
		perturbed_l, perturbed_r = [], []
		for attack in self.attacks:
			preturbed = attack.addPairNoise(image_pairs, target_labels)
			perturbed_l.append(self.resize(preturbed[0], target_size))
			perturbed_r.append(self.resize(preturbed[1], target_size))
			# perturbed_images.append(self.resize(attack.addNoise(image_pairs, target_labels), target_size))
		# return perturbed_images
		return [perturbed_l, perturbed_r]
