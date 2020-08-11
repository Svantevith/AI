import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

# Example data:
#   • An experimental drug was tested on individuals from ages 13 to 100 in a clinical trial.
#   • The trial had 2100 participants. Half were under 65 years old, half were 65 years or older.
#   • Around 95% of patients 65 or older experienced side effects.
#   • Around 95% of patients under 65 experienced no side effects.

train_samples = []
train_labels = []

# 1 - experienced side effects, 0 - did not experience any side effects
for i in range(50):
    # The ~5% of younger individuals who did experience side effects
    random_young = randint(13, 64)
    train_samples.append(random_young)
    train_labels.append(1)

    # The ~5% of older individuals who did not experience any side effects
    random_old = randint(65, 100)
    train_samples.append(random_old)
    train_labels.append(0)

for i in range(1000):
    # The 95% of younger individuals who did not experience any side effects
    random_young = randint(13, 64)
    train_samples.append(random_young)
    train_labels.append(0)

    # The 95% of older individuals who did experience side effects
    random_old = randint(65, 100)
    train_samples.append(random_old)
    train_labels.append(1)

# numpy array is the format expected in the fit function
train_samples = np.array(train_samples)
train_labels = np.array(train_labels)

train_samples, train_labels = shuffle(train_samples, train_labels)

# create feature range from 0 to 1
scaler = MinMaxScaler(feature_range=(0, 1))

# Rescale our input data from scale 13 to 100, using our scaler to range 0 to 1
# fit function does not accept 1D objects, that's why we fit_transform and reshape it into 2D

# 1D: 1 -> 2D: (-1, 1) format and the value range using scaler = (0, 1)
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1, 1))
