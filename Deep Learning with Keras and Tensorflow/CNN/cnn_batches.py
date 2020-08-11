import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.chdir('/')

# from main folder ANN & CNN/ we can define specific paths for each data set
train_path = 'data/train'
valid_path = 'data/valid'
test_path = 'data/test'

# set up batches for each data set
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(224, 224), classes=['dog', 'cat'], batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(224, 224), classes=['dog', 'cat'], batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224, 224), classes=['dog', 'cat'], batch_size=10,
                         shuffle=False)

# tf.keras.applications.vgg16.preprocess_input: what it does is, it subtracts the mean RGB value,
# computed on the training set, from each pixel.
# It computes the mean red value pixel from all of the training data,
# then that mean mean red value is subtracted from each pixel in each image.
# The same process is being repeated for each of the channel Red, Green and Blue.

# Specify shuffle=False for test_batches, because the confusion matrix will need to access the unshuffled labels.
# By default, the data sets are shuffled.


# create assertions
assert train_batches.n == 1000
assert valid_batches.n == 200
assert test_batches.n == 100
assert train_batches.num_classes == valid_batches.num_classes == test_batches.num_classes == 2
