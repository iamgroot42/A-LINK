from itertools import product, count
import numpy as np
import cv2
from cleverhans.attacks import FastGradientMethod, CarliniWagnerL2, DeepFool, ElasticNetMethod, SaliencyMapMethod, MadryEtAl, MomentumIterativeMethod, VirtualAdversarialMethod
from cleverhans.utils_keras import KerasModelWrapper


class Noise(object):
	def __init__(self, model=None, sess=None):
		self.model = model
		self.sess = sess
		pass

	def addIndividualNoise(self, image, target_labels=None):
		return image

	def addNoise(self, images, target_labels):
		noisy_images = []
		for image in images:
			noisy_images.append(self.addIndividualNoise(image, target_labels))
		return np.array(noisy_images)


class Gaussian(Noise):
	def __init__(self, mean=10, var=10, model=None, sess=None):
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
	def __init__(self, s_vs_p=0.5, amount=0.004, model=None, sess=None):
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
	def __init__(self, model=None, sess=None):
		super(Poisson, self).__init__()

	def addIndividualNoise(self, image, target_labels=None):
		vals = len(np.unique(image))
		vals = 2 ** np.ceil(np.log2(vals))
		noisy = np.random.poisson(image * vals) / float(vals)
		return noisy


class Speckle(Noise):
	def __init__(self, model=None, sess=None):
		super(Speckle, self).__init__()

	def addIndividualNoise(self, image, target_labels=None):
		row, col, ch = image.shape
		gauss = np.random.randn(row, col, ch) / 15
		gauss = gauss.reshape(row, col, ch)
		noisy = image + image * gauss
		return noisy


class Perlin(Noise):
	def __init__(self, model=None, sess=None):
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


class AdversarialNoise(Noise):
	def __init__(self, model, sess):
		super(AdversarialNoise, self).__init__(model, sess)
		self.attack_object = None
		self.attack_params = {'clip_min': 0.0, 'clip_max': 1.0}
		self.wrapped_model = KerasModelWrapper(self.model)

	def addIndividualNoise(self, image, target_labels):
		self.attack_params['batch_size'] = 1
		self.attack_params['y_target'] = target_labels
		mini_batch = [image]
		return self.attack_object.generate_np(mini_batch, **self.attack_params)[0]


class Momentum(AdversarialNoise):
	def __init__(self, model, sess):
		super(Momentum, self).__init__(model, sess)
		self.attack_object = MomentumIterativeMethod(self.wrapped_model, sess=self.sess)
		self.attack_params['eps'] = 0.3
		self.attack_params['eps_iter'] = 0.06
		self.attack_params['nb_iter'] = 3


class FGSM(AdversarialNoise):
	def __init__(self, model, sess):
		super(FGSM, self).__init__(model, sess)
		self.attack_object = FastGradientMethod(self.wrapped_model, sess=self.sess)
		self.attack_params['eps'] = 0.1


class Virtual(AdversarialNoise):
	def __init__(self, model, sess):
		super(Virtual, self).__init__(model, sess)
		self.attack_object = VirtualAdversarialMethod(self.wrapped_model, sess=self.sess)
		self.attack_params['xi'] = 1e-6
		self.attack_params['num_iterations'] = 1
		self.attack_params['eps'] = 2.0


class Madry(AdversarialNoise):
	def __init__(self, model, sess):
		super(Madry, self).__init__(model, sess)
		self.attack_object = MadryEtAl(self.wrapped_model, sess=self.sess)
		self.attack_params['nb_iter'] = 5
		self.attack_params['eps'] = 0.1


def get_relevant_noise(noise_string):
	noise_mapping = {
		'gaussian': Gaussian,
		'saltpepper': SaltPepper,
		'poisson': Poisson,
		'speckle': Speckle,
		'plain': Noise,
		'perlin': Perlin,
		'momentum': Momentum,
		'fgsm': FGSM,
		'madry': Madry
	}
	if noise_string.lower() in noise_mapping:
		return noise_mapping[noise_string.lower()]
	else:
		raise NotImplementedError("%s noise is not implemented!" % (noise_string))
	return None
