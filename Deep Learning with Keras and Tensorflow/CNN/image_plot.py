import matplotlib.pyplot as plt
import numpy as np


def plot_images(images_arr, title):
    fig, axes = plt.subplots(1, 10, figsize=(20, 20))
    axes = axes.flatten()
    for im, ax in zip(images_arr, axes):
        ax.imshow(im)
        ax.axis('off')
    plt.tight_layout()
    plt.title(title)
    plt.show()


# np.ndarray is n-dimentional array
def plot_with_labels(images, figsize=(12, 6), rows=1, interp=False, titles=None):
    if type(images[0]) is np.ndarray:
        # np.uint8 is an 8-bit unsigned integer (0 to 255)
        # we use np.ndarray to check whether our input is an image, which can be represented as multidimentional array
        # if so, we convert all of our input images into an array containing the numerical representation of each image
        images = np.array(images).astype(np.uint8)
        if images.shape[-1] != 3:
            images = images.transpose((0, 2, 3, 1))
    fig = plt.figure(figsize=figsize)
    cols = len(images) // rows if len(images) % 2 == 0 else len(images) // rows + 1
    for i in range(len(images)):
        sp = fig.add_subplot(rows, cols, i + 1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(images[i], interpolation=None if interp else 'none')
