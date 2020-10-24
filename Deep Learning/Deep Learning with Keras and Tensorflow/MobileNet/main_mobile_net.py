import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import matplotlib.pyplot as plt
from IPython.display import display, Image
from image_set import prepare_image

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# print('Num GPUs Available: ', len(physical_devices))
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# download model
mobile_model = tf.keras.applications.mobilenet.MobileNet()

im = Image(filename='data/MobileNet Samples/lizard.png', width=300, height=200)
display(im)
img = plt.imread('data\MobileNet Samples\lizard1.png')
plt.imshow(img)
plt.show()

preprocessed_image = prepare_image(filename='lizard.png')
predictions = mobile_model.predict(preprocessed_image)
# return top 5 predictions from 1000 possible image from the class
results = imagenet_utils.decode_predictions(predictions)

print(results)

assert results == results[0][0][1] == 'American chameleon'
