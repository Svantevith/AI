from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np

X, y = fetch_openml(
    'mnist_784',
    version=1,
    return_X_y=True
)

# We can see that the shape of X is 2darray (70 000, 784) meaning that we have 70000 datapoints
# with corresponding 784 features (70 000 images with res 28x28px = 784 pixels as features).
# The shape of y is 1darray containing 70 000 target values (in this case strings)
# the target can be any integer ranging from 0 to 9 (but strings!)
print(X.shape, y.shape)
print(y[:5])

# Remember that X contains numerical representation of images, let's see what values can each pixel contain
# We can see that the min val is 0 and max is 255.
# Interpreting a grayscale image, we know that 0 is white and 255 is black.
print(np.min(X), np.max(X))

# We consider only the digits '0'-'3'
# Note that to obtain both X feature matrix and y target array for digits '0'-'3',
# as a condition argument we use the target <= 'val' in both cases
# Recall that in Python you can compare '0' and '3' as numbers, while they are still being strings
X5 = X[y <= '3']
y5 = y[y <= '3']

mlp = MLPClassifier(
    hidden_layer_sizes=(6,),
    max_iter=200,
    alpha=1e-4,
    solver='sgd',
    random_state=39
)
mlp.fit(X5, y5)

# MLPClassifier stores the coefficients in the coefs_ attribute.
# We can see that it is a list with 2 elements, more specifically, with two 2d numpy arrays
print(type(mlp.coefs_))
print(len(mlp.coefs_))
# The two elements (2darrays) in the coefs_ attribute correspond to 2 layers with trainable parameters:
# the hidden and output layer
# The layer 0 in coefs_ list is the hidden layer with shape (784, 6) meaning that we have 6 nodes,
# and each node has exactly 784 inputs

# So in general, we have 70 000 datapoints (images), each 28x28px finally containing 784 pixels (features).
# All of 784 features are forming the Input Layer and these 784 pixels (features) are being an input
# to the 6 nodes in the Hidden Layer. Then the output layer contains 3 possible class labels and
# each of these 3 Output nodes contain 6 inputs.
print(mlp.coefs_[0].shape)

fig, axes = plt.subplots(2, 3, figsize=(5, 4))
for i, ax in enumerate(axes.ravel()):
    # Remember that coef_ attribute is a list containing 2darrays as elements
    # Here we focus only on the 0th coef_ element which is the Hidden Layer
    # which is represented as a 2darray. The shape of it is (784, 6)
    # each of 6 nodes contains 784 features,
    # so we want to have all features and only specific i-node (digit) [:, i]
    coef = mlp.coefs_[0][:, i]
    ax.matshow(
        coef.reshape(28, 28),
        cmap=plt.cm.gray
    )
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(i + 1)
plt.show()

# For example to print the only first node of the hidden layer:
# Choose the coefs_ attribute 0th 2darray corresponding to hidden layer
# Then choose all of the inputs (features) to that node
# Choose node number you would like to get results to
node_1 = mlp.coefs_[0][:, 0]
plt.matshow(
    node_1.reshape(28, 28),
    cmap=plt.cm.gray
)
plt.xticks(())
plt.yticks(())
plt.title('First node in the hidden layer')
plt.tight_layout(False)
plt.show()

# The Computation of Neural Networks:
# ANNs can take a decent amount of time to train. Each node contains its own coefficients and to train they are
# iterativelu updated, so this can be time consuming. However they are parallelizable, so it is possible to throw
# computer power at them to make them train faster. Once the Neural Networks are built, they are not slow
# to make predictions, but still they are not the fastest in comparison to other models.

# The Performance of Neural Networks is very hard to beat by other models. The ANNs can take some tuning parameters
# to find the optimal performance, however they benefit from needing minimal feature engineering prior to
# building the model. For smaller and more structured datasets it is encouraged to use simpler models like
# Logistic Regression and achieve still great performance, however in case of large unstructured datasets,
# Neural Networks outperform other models.
