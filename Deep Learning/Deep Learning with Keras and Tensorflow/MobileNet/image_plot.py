import matplotlib.pyplot as plt


def plot_images(images_arr, title):
    fig, axes = plt.subplots(1, 10, figsize=(20, 20))
    axes = axes.flatten()
    for im, ax in zip(images_arr, axes):
        ax.imshow(im)
        ax.axis('off')
    plt.tight_layout()
    plt.title(title)
    plt.show()
