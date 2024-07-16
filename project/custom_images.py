from PIL import Image
import numpy as np

# def convert_to_binary(image_path, thr: float = 0.5):
#     # Load the image
#     gray_image = np.array(Image.open(image_path).convert('L'))
#     binary_image = gray_image > thr * gray_image.max()
#     return binary_image


def convert_to_binary(image_path, max_density: float = 0.5):
    # Load the image
    gray_image = np.array(Image.open(image_path).convert('L'))

    # Turn into a density function
    prob = gray_image / 255 * max_density

    # Randomly sample from it
    binary_image = np.random.uniform(low=0, high=1, size=gray_image.shape) < prob

    return binary_image, prob