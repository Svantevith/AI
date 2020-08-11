# Neural Networks are incredibly popular and powerful machine learning models. They often perform well in cases where
# we have a lot of features as they automatically do feature engineering without requiring domain knowledge to
# restructure the features.

# In this module we will be using image data. Since each pixel in the image is a feature,
# we can have a really large feature set. They are all commonly used in text data as it has a large feature set as
# well. Voice recognition is another example where neural networks often shine.

# Neural networks often work well
# without you needing to use domain knowledge to do any feature engineering.

# Each neuron is only capable of a small computation, but when working together they become capable of solving large
# and complicated problems.

# Inside the neuron, to do the computation to produce the output, we first put the inputs into the following equation
# (just like in logistic regression): w1x1 + w2x2 + b
# Recall that x1 and x2 are the inputs. In logistic regression, we referred to the values w1, w2, and b
# as the coefficients. In neural networks, we refer to w1 and w2 as the weights, and b as the bias.

# We plug this value into what is called an activation function. The above equation can have a result of any real
# number. The activation function condenses it into a fixed range (often between 0 and 1).

# To get the output from the inputs we do the following computation. The weights, w1 and w2, and the bias, b,
# control what the neuron does. We call these values (w1, w2, b) the parameters. The function f is the activation
# function (in this case the sigmoid function). The value y is the neuron’s output.
# y = activation_function(w1x1 + w2x2 + b)
# This function can be generalized to have any number of inputs (xi) and thus the corresponding number of weights (wi).

# To create a neural network we combine neurons together so that the outputs of some neurons are inputs of other
# neurons. We will be working with feed forward neural networks which means that the neurons only send signals in one
# direction. In particular, we will be working with what is called a Multi-Layer Perceptron (MLP). The neural network
# has multiple layers which we see depicted below.

# A single-layer perceptron is a neural network without any hidden layers. These are rarely used. Most neural
# networks are multi-layer perceptrons, generally with one or two hidden layers.

from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import numpy as np

X, y = make_classification(
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=17
)

# print('Feature matrix:\n', X)
# print('Target array:\n', y)

# because it is a 2D matrix,[y==0] and [y==1] work as keys for respective classes in feature matrix X, while [:,
# 0] and [:, 1] show all rows respective for 1st and 2nd features (columns)

# scatter plot for class0 X[y==0], showing all rows for first and second feature
plt.scatter(
    X[y == 0][:, 0],
    X[y == 0][:, 1],
    s=100,
    edgecolors='k'
)

# scatter plot for class1 X[y==1], showing all rows for dirst and second feature
plt.scatter(
    X[y == 1][:, 0],
    X[y == 1][:, 1],
    s=100,
    edgecolors='k',
    marker='^'
)

plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33)

ann = MLPClassifier(
    max_iter=1000,
    random_state=39
)
ann.fit(X_train, y_train)
print('Accuracy of default MLP Classifier:', ann.score(X_test, y_test))

# In general, the more data you have, the fewer iterations you need to converge. If the value is too large,
# it will take too long to run the code. If the value is too small, the neural network will not converge on the
# optimal solution.

# We also sometimes need to change alpha (L2 penalty parameter), which is the step size. This is how much the neural
# network changes the coefficients at each iteration. If the value is too small, you may never converge on the
# optimal solution. If the value is too large, you may miss the optimal solution. Initially you can leave this at the
# default. The default value of alpha is 0.0001. Note that decreasing alpha often requires an increase in max_iter.

# Sometimes you will want to change the solver. This is what algorithm is used to find the optimal solution. All the
# solvers will work, but you may find for your dataset that a different solver finds the optimal solution faster. The
# options for solver are 'lbfgs', 'sgd' and 'adam' - the solvers are used to minimize to loss function and to update
# the trainable parameters.

tuned_ann = MLPClassifier(
    max_iter=1000,
    hidden_layer_sizes=(100, 50),
    activation='relu',
    alpha=0.0001,
    solver='adam',
    random_state=39
)
tuned_ann.fit(X_train, y_train)
print('Accuracy of tuned MLP Classifier:', tuned_ann.score(X_test, y_test), '\n')

# Predicting the output for 2 digits
# ###################################

# n_class is 2 because we will only classify image of digits 0 and 1
X, y = load_digits(
    n_class=2,
    return_X_y=True
)

# The shape of the feature matrix is 360x64, meaning 360 input images, each having total of 64 features (8x8px)

# The shape of target array is a 1D array containing 360 labels for corresponding images
print(X.shape, y.shape)
print('\nNumeric representation of the first image:\n', X[0])
print('Label of the first image:', y[0])

# Let's reshape the numeric representation of the first image, so we can view it as 8x8 matrix
print('\nNumeric representation of the first image after reshaping it to (8, 8):\n', X[0].reshape(8, 8).astype(int))


# plot the images using the values computed from each pixel of their reshaped (n x m pixels) matrices using matshow
# method
def draw_img(x: np.ndarray) -> None:
    plt.matshow(
        x,
        cmap=plt.cm.gray
    )
    plt.title('Image obtained using matshow (obtained from the matrix containing values for each pixel)')
    # remove axes ticks
    plt.xticks(
        ()
    )
    plt.yticks(
        ()
    )
    plt.show()


show_image = draw_img

show_image(X[0].reshape(8, 8).astype(int))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=22)

mlp = MLPClassifier(random_state=39)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
print('\nPrediction for first image:', y_pred[0])
print('\nReal class:', y[0])
print('\nFirst Image:')
show_image(X_test[0].reshape(8, 8).astype(int))
print('\nAccuracy of MLP for dataset containing 2 digits:', mlp.score(X_test, y_test))

# Predicting the output for all digits
# ###################################

X, y = load_digits(
    n_class=10,
    return_X_y=True
)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=22)

mlp.hidden_layer_sizes = (100, 100, 100)
mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)
print('\nAccuracy of MLP for dataset containing 10 digits:', mlp.score(X_test, y_test))

# 2D numpy array containing the numerical representation of all wrong predictions

# feature matrix containing numerical representation of each incorrectly classified digits
incorrect = X_test[y_pred != y_test]

# true class labels for incorrect predictions
incorrect_true = y_test[y_pred != y_test]

# incorrect predictions
incorrect_pred = y_pred[y_pred != y_test]

print('\nNumerical representation of the first incorrectly predicted digit:\n', incorrect[0].reshape(8, 8).astype(int))

print('\nTrue class label for incorrect prediction:', incorrect_true[0])

print('Incorrect prediction:', incorrect_pred[0])
