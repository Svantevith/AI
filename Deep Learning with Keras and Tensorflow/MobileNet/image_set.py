import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image


def prepare_image(filename):
    os.chdir('D:\PyCharm Professional\Projects\Deep Learning\MobileNet')
    img_path = 'data/MobileNet Samples/'
    img = image.load_img(img_path + filename, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_with_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_with_expanded_dims)
