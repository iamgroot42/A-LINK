import numpy as np 
import cv2


class Noise:
	def __init__(self, model=None):
		self.model = model
		pass

	def addIndividualNoise(self, image):
		return image

	def addNoise(self, images):
		noisy_images = []
		for image in images:
			noisy_images.append(self.addIndividualNoise(image))
		return np.array(noisy_images)


class Gaussian(Noise):
	def __init__(self, mean=0, var=0.1):
		super(Gaussian, self).__init__()
		self.mean = mean
		self.var = var
		self.sigma = self.var ** 0.5

	def addIndividualNoise(self, image):
		row, col, ch= image.shape
		gauss = np.random.normal(self.mean, self.sigma, (row, col, ch))
		gauss = gauss.reshape(row, col, ch)
		noisy = image + gauss
		return noisy


class SaltPepper(Noise):
	def __init__(self, s_vs_p=0.5, amount=0.004):
		super(SaltPepper, self).__init__()
		self.s_vs_p = s_vs_p
		self.amount = amount

	def addIndividualNoise(self, image):
		row, col, ch= image.shape
		out = np.copy(image)
		# Salt mode
		num_salt = np.ceil(self.amount * image.size * self.s_vs_p)
		coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
	      	out[coords] = 1
		# Pepper mode
		num_pepper = np.ceil(self.amount* image.size * (1. - self.s_vs_p))
		coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
		out[coords] = 0
		return out


class Poisson(Noise):
	def __init__(self):
		super(Poisson, self).__init__()

	def addIndividualNoise(self, image):
		vals = len(np.unique(image))
		vals = 2 ** np.ceil(np.log2(vals))
		noisy = np.random.poisson(image * vals) / float(vals)
		return noisy


class Speckle(Noise):
	def __init__(self):
		super(Speckle, self).__init__()

	def addIndividualNoise(self, image):
		row, col, ch = image.shape
		gauss = np.random.randn(row, col, ch)
		gauss = gauss.reshape(row, col, ch)        
		noisy = image + image * gauss
		return noisy
