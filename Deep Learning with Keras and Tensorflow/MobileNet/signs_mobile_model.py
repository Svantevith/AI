import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_path = 'data/MobileNet Samples/Sign Language Samples/train'
valid_path = 'data/MobileNet Samples/Sign Language Samples/valid'
test_path = 'data/MobileNet Samples/Sign Language Samples/test'

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)\
    .flow_from_directory(directory=train_path, target_size=(224, 224), batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)\
    .flow_from_directory(directory=valid_path, target_size=(224, 224), batch_size=10)
# Remember to set shuffle = False in the test set !
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)\
    .flow_from_directory(directory=test_path, target_size=(224, 224), batch_size=10, shuffle=False)

assert train_batches.n == 1712
assert valid_batches.n == 300
assert test_batches == 50
assert train_batches.num_classes == valid_batches.num_classes == test_batches.num_classes == 10
