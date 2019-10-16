import numpy as np
from differential_evolution import differential_evolution


def perturb_image(xs, img):
    # If this function is passed just one perturbation vector,
    # pack it in a list to keep the computation the same
    if xs.ndim < 2:
        xs = np.array([xs])

    # Copy the image n == len(xs) times so that we can 
    # create n new perturbed images
    tile = [len(xs)] + [1] * (xs.ndim + 1)
    imgs = np.tile(img, tile)

    # Make sure to floor the members of xs as int types
    xs = xs.astype(int)

    for x, img in zip(xs, imgs):
        # Split x into an array of 5-tuples (perturbation pixels)
        # i.e., [[x,y,r,g,b], ...]
        pixels = np.split(x, len(x) // 5)
        for pixel in pixels:
            # At each pixel's x,y position, assign its rgb value
            # x_pos, y_pos, *rgb = pixel
            x_pos, y_pos, r, g, b = pixel
            img[x_pos, y_pos] = [r, g, b]

    return imgs


class PixelAttacker:
    def __init__(self, model):
        # Load data and model
        self.model = model

    def predict_classes(self, xs, img, target_class, minimize=True):
        # Perturb the image with the given pixel(s) x and get the prediction of the model
        imgs_perturbed = perturb_image(xs, img)
        predictions = self.model.predict(imgs_perturbed)[:, target_class]
        # This function should always be minimized, so return its complement if needed
        return predictions if minimize else 1 - predictions

    def attack_success(self, x, img, target_class, targeted_attack=False, verbose=False):
        # Perturb the image with the given pixel(s) and get the prediction of the model
        attack_image = perturb_image(x, img)

        confidence = self.model.predict(attack_image)[0]
        predicted_class = np.argmax(confidence)

        # If the prediction is what we want (misclassification or 
        # targeted classification), return True
        if verbose:
            print('Confidence:', confidence[target_class])
        if ((targeted_attack and predicted_class == target_class) or
                (not targeted_attack and predicted_class != target_class)):
            return True

    def attack(self, image, actual_class, target, pixel_count, dimensions,
               maxiter=75, popsize=400, verbose=False):
        # Change the target class based on whether this is a targeted attack or not
        targeted_attack = target is not None
        target_class = target if targeted_attack else actual_class

        # Define bounds for a flat vector of x,y,r,g,b values
        # For more pixels, repeat this layout
        dim_x, dim_y = dimensions
        bounds = [(0, dim_x), (0, dim_y), (0, 256), (0, 256), (0, 256)] * pixel_count

        # Population multiplier, in terms of the size of the perturbation vector x
        popmul = max(1, popsize // len(bounds))

        # Format the predict/callback functions for the differential evolution algorithm
        def predict_fn(xs):
            return self.predict_classes(xs, image, target_class, target is None)

        def callback_fn(x, convergence):
            return self.attack_success(x, image, target_class, targeted_attack, verbose)

        # Call Scipy's Implementation of Differential Evolution
        attack_result = differential_evolution(
            predict_fn, bounds, maxiter=maxiter, popsize=popmul,
            recombination=1, atol=-1, callback=callback_fn, polish=False)

        # Calculate some useful statistics to return from this function
        attack_image = perturb_image(attack_result.x, image)[0]
        predicted_probs = self.model.predict(np.array([attack_image]))[0]

        return attack_image

    def attack_all(self, input_data, targets, dimensions, pixel_count=40, maxiter=50, popsize=250, verbose=False):
        X = []

        for i, img in enumerate(input_data):

            target_class = np.argmax(targets[i])
            result = self.attack(img, 1 - target_class, target_class,
                                    pixel_count, dimensions, 
                                    maxiter=maxiter, popsize=popsize,
                                    verbose=verbose)
            X.append(result)

        return X
