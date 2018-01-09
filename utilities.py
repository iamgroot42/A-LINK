import cv2


def resize(images, new_size):
	resized_images = []
	for image in images:
		resized_images.append(cv2.resize(image, new_size))
	return np.array(resized_images)
