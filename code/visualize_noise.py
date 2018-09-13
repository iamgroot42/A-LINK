import noise
import numpy as np
import sys
from PIL import Image
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

img = cv2.resize(np.asarray(Image.open(sys.argv[1]).convert('RGB'), dtype=np.float32), (224,224))
perlin = noise.get_relevant_noise("perlin")()
noisy_image = perlin.addIndividualNoise(img)
plt.imshow(noisy_image/255.)
plt.savefig("lol.png")
