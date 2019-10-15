from itertools import product, count
import numpy as np
import cv2
from cleverhans.attacks import FastGradientMethod, CarliniWagnerL2, DeepFool, ElasticNetMethod, SaliencyMapMethod, MadryEtAl, MomentumIterativeMethod, VirtualAdversarialMethod
from cleverhans.utils_keras import KerasModelWrapper
from keras.models import Model, Sequential
from keras.layers import Input, Lambda, Activation
from keras import backend as K


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


class AdversarialNoise(Noise):
	def __init__(self, model, sess, feature_model):
		super(AdversarialNoise, self).__init__(model, sess, feature_model)
		self.attack_object = None
		self.attack_params = {'clip_min': 0.0, 'clip_max': 255.0}

	# Concatenate featurization and siamese network to create end to end differentiable model (for attack)
	# Attack modifies both images together
	# Currently supports only Keras models (no arcnet featurization model)
	def get_e2e_model(self):
		# Locate Lambda layer
		lambda_layer = 0
		for i, layer in enumerate(self.model.siamese_net.layers):
			if "lambda_" in layer.name:
				lambda_layer = i
				break
		
		# Concatenate images along rows
		new_input  = Input((2 * self.model.shape[0],) + self.model.shape[1:])
		left_half  = Lambda(lambda x: x[:self.model.shape[0]])(new_input)
		right_half = Lambda(lambda x: x[self.model.shape[0]:])(new_input)

		left_feature  = self.feature_model.model(left_half)
		right_feature = self.feature_model.model(right_half)

		L1_layer      = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
		siamese_input = L1_layer([left_feature, right_feature])

		# Clone dense layers of siamese network into new model (super hacky, not proud of it but hey it works!)
		final_output = siamese_input
		dense_clone_layers = self.model.getDenseBarebones()
		for layer in dense_clone_layers:
			final_output = layer(final_output)
		# Add softmax layer
		final_output = Activation('softmax')(final_output)
		# cleverhans-ready model
		model_for_adversary = Model(new_input, final_output)
		# Copy weights for last dense layers
		for i in range(len(dense_clone_layers)):
			model_for_adversary.layers[i - (len(dense_clone_layers) + 1)].set_weights(self.model.siamese_net.layers[lambda_layer + 1 + i].get_weights())
		return model_for_adversary

	def addPairNoise(self, image_pairs, target_labels):
		left_half, right_half = [], []
		
		# Create wrappers, ready attack object
		e2e_model = self.get_e2e_model(r_image)
		wrapped_model = KerasModelWrapper(e2e_model)
		attack_object = self.attack_object(wrapped_model, sess=self.sess)

		attack_params = self.attack_params.copy()
		attack_params['y_target'] = target_labels
		concat_data = np.array([np.concatenate((l, r), axis=0) for (l,r) in image_pairs])
		perturbed = attack_object.generate_np(concat_data, **attack_params)[0]
		left_half  = [p[:image_pairs[0].shape[0]] for p in perturbed]
		right_half = [p[image_pairs[0].shape[0]:] for p in perturbed]
		return [left_half, right_half]


class Momentum(AdversarialNoise):
	def __init__(self, model, sess, feature_model):
		super(Momentum, self).__init__(model, sess, feature_model)
		self.attack_object = MomentumIterativeMethod #(self.wrapped_model, sess=self.sess)
		self.attack_params['eps'] = 0.3
		self.attack_params['eps_iter'] = 0.06
		self.attack_params['nb_iter'] = 3


class FGSM(AdversarialNoise):
	def __init__(self, model, sess, feature_model):
		super(FGSM, self).__init__(model, sess, feature_model)
		self.attack_object = FastGradientMethod
		self.attack_params['eps'] = 10.0


class Virtual(AdversarialNoise):
	def __init__(self, model, sess, feature_model):
		super(Virtual, self).__init__(model, sess, feature_model)
		self.attack_object = VirtualAdversarialMethod
		self.attack_params['xi'] = 1e-6
		self.attack_params['num_iterations'] = 1
		self.attack_params['eps'] = 2.0


class Madry(AdversarialNoise):
	def __init__(self, model, sess, feature_model):
		super(Madry, self).__init__(model, sess, feature_model)
		self.attack_object = MadryEtAl
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

