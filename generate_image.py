import numpy as np

# For headless machines
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('path', 'ADX.npy', 'Path where adversarial examples are saved')
flags.DEFINE_integer('example_index', 0, 'Which index do you want to visualize?')

def main(argv=None):

	X_test_adv = np.load(FLAGS.path)
	X_test_adv = X_test_adv[FLAGS.example_index]

	# Infer type of image from data
	if X_test_adv.shape[0] == 1:
		plt.matshow(X_test_adv[0],  cmap='gray')
		plt.savefig('adv_example.png')
	else:
		#X_test_adv = np.swapaxes(X_test_adv,0,2)
		#X_test_adv = np.swapaxes(X_test_adv,0,1)
		X_test_adv = (X_test_adv).astype('uint8')
		plt.imshow(X_test_adv)
		plt.savefig('adv_example.png')
	print("Image saved!")


if __name__ == '__main__':
	app.run()
