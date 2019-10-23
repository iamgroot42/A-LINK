from itertools import product, count
import numpy as np
import cv2
from keras.models import Model, Sequential, clone_model
from keras.layers import Input, Lambda, Activation
from keras import backend as K
import attack


class Noise(object):
	def __init__(self, model=None, sess=None, feature_model=None):
		self.model = model
		self.sess = sess
		self.feature_model = feature_model
		pass

	def addIndividualNoise(self, image, target_labels=None):
		return image

	def addNoise(self, images, target_labels):
		noisy_images = []
		for i, image in enumerate(images):
			noisy_images.append(self.addIndividualNoise(image, target_labels[i]))
		return np.array(noisy_images)

	def addPairNoise(self, image_pairs, target_labels):
		noisy_images = []
		left_half  = self.addNoise(image_pairs[0], target_labels)
		right_half = self.addNoise(image_pairs[1], target_labels)
		return [left_half, right_half]


class Gaussian(Noise):
	def __init__(self, mean=10, var=10, model=None, sess=None, feature_model=None):
		super(Gaussian, self).__init__()
		self.mean = mean
		self.var = var
		self.sigma = self.var ** 0.5

	def addIndividualNoise(self, image, target_labels=None):
		row, col, ch= image.shape
		gauss = np.random.normal(self.mean, self.sigma, (row, col, ch))
		gauss = gauss.reshape(row, col, ch)
		noisy = image + gauss
		return noisy


class SaltPepper(Noise):
	def __init__(self, s_vs_p=0.5, amount=0.004, model=None, sess=None, feature_model=None):
		super(SaltPepper, self).__init__()
		self.s_vs_p = s_vs_p
		self.amount = amount

	def addIndividualNoise(self, image, target_labels=None):
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
	def __init__(self, model=None, sess=None, feature_model=None):
		super(Poisson, self).__init__()

	def addIndividualNoise(self, image, target_labels=None):
		vals = len(np.unique(image))
		vals = 2 ** np.ceil(np.log2(vals))
		noisy = np.random.poisson(image * vals) / float(vals)
		return noisy


class Speckle(Noise):
	def __init__(self, model=None, sess=None, feature_model=None):
		super(Speckle, self).__init__()

	def addIndividualNoise(self, image, target_labels=None):
		row, col, ch = image.shape
		gauss = np.random.randn(row, col, ch) / 15
		gauss = gauss.reshape(row, col, ch)
		noisy = image + image * gauss
		return noisy


class Perlin(Noise):
	def __init__(self, model=None, sess=None, feature_model=None):
		super(Perlin, self).__init__()

	def individualFilterNoise(self, size, ns):
		nc = int(size / ns)  # number of nodes
		grid_size = int(size / ns + 1)  # number of points in grid

		def generate_unit_vectors(n):
			'Generates matrix NxN of unit length vectors'
			phi = np.random.uniform(0, 2*np.pi, (n, n))
			v = np.stack((np.cos(phi), np.sin(phi)), axis=-1)
			return v

		# generate grid of vectors
		v = generate_unit_vectors(grid_size)

		# generate some constans in advance
		ad, ar = np.arange(ns), np.arange(-ns, 0, 1)

		# vectors from each of the 4 nearest nodes to a point in the NSxNS patch
		vd = np.zeros((ns, ns, 4, 1, 2))
		for (l1, l2), c in zip(product((ad, ar), repeat=2), count()):
			vd[:, :, c, 0] = np.stack(np.meshgrid(l2, l1, indexing='xy'), axis=2)

		# quintic interpolation
		qz = lambda t: t * t * t * (t * (t * 6 - 15) + 10)

		# interpolation coefficients
		d = qz(np.stack((np.zeros((ns, ns, 2)),
						 np.stack(np.meshgrid(ad, ad, indexing='ij'), axis=2)), axis=2) / ns)
		d[:, :, 0] = 1 - d[:, :, 1]
		# make copy and reshape for convenience
		d0 = d[..., 0].copy().reshape(ns, ns, 1, 2)
		d1 = d[..., 1].copy().reshape(ns, ns, 2, 1)

		# make an empy matrix
		m = np.zeros((size, size))
		# reshape for convenience
		t = m.reshape(nc, ns, nc, ns)

		# calculate values for a NSxNS patch at a time
		for i, j in product(np.arange(nc), repeat=2):  # loop through the grid
			# get four node vectors
			av = v[i:i+2, j:j+2].reshape(4, 2, 1)
			# 'vector from node to point' dot 'node vector'
			at = np.matmul(vd, av).reshape(ns, ns, 2, 2)
			# horizontal and vertical interpolation
			t[i, :, j, :] = np.matmul(np.matmul(d0, at), d1).reshape(ns, ns)
		return m

	def addIndividualNoise(self, image, target_labels=None):
		row, col, ch = image.shape
		assert(row == col)
		if row % 56 == 0:
			noise = np.sum([self.individualFilterNoise(row, i) for i in [56, 32, 16]], axis=0)
		else:
			noise = np.sum([self.individualFilterNoise(row, i) for i in [50, 30, 15]], axis=0) 
		noisy = image + np.repeat(np.expand_dims(noise, 2), 3, 2)
		return noisy


class PredictionWrappedModel:
	def __init__(self, model, feature_model):
		self.model = model
		self.feature_model = feature_model

	def predict(self, X):
		# Split into left and right halves
		left_half  = [p[:X[0].shape[0]/2] for p in X]
		right_half = [p[X[0].shape[0]/2:] for p in X]
		if self.feature_model:
			left_features  = self.feature_model.process(left_half)
			right_features = self.feature_model.process(right_half)
		else:
			left_features  = left_half
			right_features = right_half
		return self.model.predict([left_features, right_features])


class AdversarialNoise(Noise):
	def __init__(self, model, sess, feature_model):
		super(AdversarialNoise, self).__init__(model, sess, feature_model)
		# Create wrappers, ready attack object
		self.e2e_model = PredictionWrappedModel(model, feature_model)
		self.attacker = attack.PixelAttacker(self.e2e_model)

	def addPairNoise(self, image_pairs, target_labels):
		left_half, right_half = [], []
		
		concat_data = [np.concatenate((image_pairs[0][i], image_pairs[1][i]), axis=0) for i in range(len(image_pairs[0]))]
		img_shape  = image_pairs[0][0].shape
		perturbed  = self.attacker.attack_all(concat_data, target_labels, dimensions=(2*img_shape[0], img_shape[1]))

		left_half  = [p[:p.shape[0]/2] for p in perturbed]
		right_half = [p[p.shape[0]/2:] for p in perturbed]

		return [left_half, right_half]


def get_relevant_noise(noise_string):
	noise_mapping = {
		'gaussian': Gaussian,
		'saltpepper': SaltPepper,
		'poisson': Poisson,
		'speckle': Speckle,
		'plain': Noise,
		'perlin': Perlin,
		'adversarial': AdversarialNoise,
	}
	if noise_string.lower() in noise_mapping:
		return noise_mapping[noise_string.lower()]
	else:
		raise NotImplementedError("%s noise is not implemented!" % (noise_string))
	return None
