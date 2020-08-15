# n the last module, we built a linear regression model to predict a continuous value,
# the median home value in Boston. In this module, we will work through classification problems,
# whose task is to predict a discrete value.

# Discrete data are only able to have certain values, while continuous data can take on any value.
# Examples of classification problems involving discrete data values are:
# • to predict whether a breast cancer is benign or malignant given a set of features
# • to classify an image as containing cats or dogs or horses
# • to predict whether an email is spam or not from a given email address
# In each of the examples, the labels come in categorical form and represent a finite number of classes.
# In regression model we predict the continuous values, which are in numerical form and can take any number of values.

# Discrete data values can be numeric, like the number of students in a class,
# or it can be categorical, like red, blue or yellow, while containing finite number of classes.

# Discrete data can only take particular values. Continuous data are not restricted to defined separate values,
# but can occupy any value over a continuous range. Between any two continuous data values,
# there may be an infinite number of others. Continuous data are always essentially numeric.

# There are two types of classification: binary and multi-class. If there are two classes to predict,
# that is a binary classification problem, for example, a benign or malignant tumor. When there are more than two
# classes, the task is a multi-classification problem. For example, classifying the species of iris, which can be
# versicolor, virqinica, or setosa, based on their sepal and petal characteristics. Common algorithms for
# classification include logistic regression, k nearest neighbors, decision trees, naive bayes, support vector
# machines, neural networks, etc.
# Here we will learn how to use k nearest neighbors to classify iris species.

# Supervised learning problems are grouped into regression and classification problems. Both problems have as a goal
# the construction of a mapping function from input variables (X) to an output variable (y). The difference is that
# the output variable is continuous & numerical in regression and discrete & categorical for classification.

# Machine Learning
# 1. Supervised Learning
#       a) Regression
#           i) Linear Regression
#           ii) Polynomial Regression
#           iii) Decision Tree Regression
#           iv) Random Forest Regression
#       b) Classification
#           i) Logistic Regression
#           ii) K-nearest neighbors
#           iii) Support Vector Machines (SVM)
#           iv) Naive Bayes Classifier
#           v) Kernel SVM
#           vi) Decision Tree Classifier
#           vii) Random Forest Classifier
#           viii) Neural Network Classifiers (Multi Layer Perceptron)
# 2. Unsupervised Learning
#       a) Clustering
#            i) k-means clustering
#            ii) Hierarchical clustering
#            iii) Density-based clustering
#            iv) Fuzzy clustering
#         b) Dimensionality Reduction
#             i) Principle Component Analysis
#             ii) Kernel Principle Component Analysis
# 3. Semi-supervised Learning
# 4. Reinforcement Learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, plot_confusion_matrix, precision_recall_fscore_support, confusion_matrix

# we may use index_col='species' to group the dataset by 'species' column
# the 'species' would be used as an index
# by default the index_col are just integers (indices) in the range from 0 to number of rows of the dataset
# Recall that read_csv returns pandas DataFrame
iris_df = pd.read_csv('data/iris.csv')
print('Shape of Iris dataset', iris_df.shape)
print(iris_df)

# Sometimes we get the column which we do not want to have (not informative, thus redundant)
# we can delete this column using code:
# dataset.drop('column_name', axis=1, inplace=True)
# axis=1 means that we delete the column (horizontal stacking, by column, axis=1),
# and inplace=True ensures that remaining columns will not be shuffled

# Recall:
# Series is a one-dimensional array of values. Series object has only “axis 0” because it has only one dimension.
# Usually, in Python, one-dimensional structures are displayed as a row of values.
# On the contrary, here we see that Series is displayed as a column of values.
# Each cell in Series is accessible via index value along the “axis 0”.
# Axis 0 is a vertical axis representing rows.
# Remember vertical stacking (by rows, axis=0) and horizontal stacking (by column, axis=1)

# The ranges of attributes are still of similar magnitude, thus we will skip standardization. However,
# standardizing attributes such that each has a mean of zero and a standard deviation of one,
# can be an important preprocessing step for many machine learning algorithms.
print('Iris dataset Summary Statistics', iris_df.describe())

# As given in the yellow box, Feature scaling is very crucial if your features have different ranges.
# Ex-Range of Feature 1- (2-50)
# Range of Feature 2- (0.002-0.1)
# Range of Feature 3- (1000-20000)
#
# Then, make sure to either user StandardScaler() or RobustScaler() after train_test_split() before initializing and
# training your model.
# If feature scaling is not done, it will take a lot of time to find the optimized parameters for the model of your
# interest and the weight(slope of that feature) of the highest range feature(in this example - Feature 3) will be
# much more than that of other 2 features. So basically, your model will depend heavily on the value of Feature 3,
# and negligible dependence on the values of Feature 1 and 2 will be seen. This will lead to skewed predictions.
# https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html
# Take Home Message - Do feature scaling. It will never do any harm ;)

#######################################################################################################################

# The columns in the DataFrame contain feature_names of the iris dataset
# We only need columns containing numerical values
numeric_cols = list(
    iris_df.select_dtypes(
        include=['int64', 'float64']
    ).columns
)
print(numeric_cols)

# Standard Scaler We will start our little tour (not exhaustive) of scaling techniques by probably the least risky:
# StandardScaler (). This technique assumes that data is normally distributed. The function will recalculate each
# characteristic so that the data gets centered around 0 and 1
# z = x - mean / std
from sklearn.preprocessing import StandardScaler

st_scaler = StandardScaler()
st_scaled_iris_df = st_scaler.fit_transform(iris_df[numeric_cols])
st_scaled_iris_df = pd.DataFrame(st_scaled_iris_df, columns=numeric_cols)

# So the standardization removes the mean and scales the data to unit variance. However, the outliers still have an
# influence when computing the empirical mean and standard deviation.

# MinMaxScaler
# This technique transforms each feature (x) by adapting it on a given range (by default [0, 1]). It is possible to
# change this range via the parameters feature_range = tuple (min, max). To make it simple, here is the
# transformation formula for each feature:
# xscaled = x - xmin / xmax - xmin
from sklearn.preprocessing import MinMaxScaler

mm_scaler = MinMaxScaler()
mm_scaled_iris_df = mm_scaler.fit_transform(iris_df[numeric_cols])
mm_scaled_iris_df = pd.DataFrame(mm_scaled_iris_df, columns=numeric_cols)

# If this technique using MinMaxScaler is probably the best known, it works especially well for cases where the
# distribution is not Gaussian or when the Standard Deviation is small. However, MinMaxScaler() is very sensitive to
# outliers. In this case, we quickly switch to the last technique: RobustScaler()

# MaxAbsScaler
# This scaling technique is useful when the distribution of values ​​is sparse and you have quite a few integers.
# The absolute values are mapped in the range [0, 1]. Indeed, on positive only data, this scaler behaves similarly
# to MinMaxScaler () and therefore also suffers from the presence of large outliers.
from sklearn.preprocessing import MaxAbsScaler

ma_scaler = MaxAbsScaler()
ma_scaled_iris_df = ma_scaler.fit_transform(iris_df[numeric_cols])
ma_scaled_iris_df = pd.DataFrame(ma_scaled_iris_df, columns=numeric_cols)

# RobustScaler The RobustScaler () technique uses the same principle of scaling as MinMaxScaler (). However,
# it uses the interquartile range instead of the min-max, which makes it more reliable with regard to outliers. Here
# is the formula for re-working the features:
# xi - 1st quartile (25%) / 3rd quartile (75%) - 1st Quartile (25%)
from sklearn.preprocessing import RobustScaler

rb_scaler = RobustScaler()
rb_scaled_iris_df = rb_scaler.fit_transform(iris_df[numeric_cols])
rb_scaled_iris_df = pd.DataFrame(rb_scaled_iris_df, columns=numeric_cols)

# Unlike the previous scalers, this scaler uses some centering and scaling statistics which are based on percentiles.
# Therefore they are not influenced by very large marginal outliers. Consequently, the resulting range of the
# transformed feature values is larger than for the previous scalers and, more importantly, are approximately similar.

# Let’s just summarize the Feature Scaling techniques we just encountered:
# 1) Scaling features to a range, often between zero and one, can be achieved using MinMaxScaler or MaxAbsScaler.
# 2) MaxAbsScaler was specifically designed for scaling sparse data, RobustScaler cannot be fitted to sparse inputs,
# but you can use the transform method on sparse inputs.
# 3) If your data contains many outliers, scaling using the mean and variance of the data is likely to not work
# very well. In this case, you need to use RobustScaler instead.

#######################################################################################################################

# We can see that the dataset is equally distributed into 3 classes, each containing 50 samples.
# We can check it using the groupby('column').size() or df['column'].values_counts() method.
# The method value_counts() is a great utility for quickly understanding the distribution of the data. When used on
# the categorical data, it counts the number of unique values in the column of interest.

# Iris is a balanced dataset as the data points for each class are evenly distributed.

# An example of an imbalanced dataset is fraud. Generally only a small percentage of the total number of transactions
# is actual fraud, about 1 in 1000. And when the dataset is imbalanced, a slightly different analysis will be used.
# Therefore, it is important to understand whether the data is balanced or imbalanced.
# An imbalanced dataset is one where the classes within the data are not equally represented.
print(iris_df.groupby('species').size())  # returns size of each of the class in the species column (50)
print(iris_df['species'].value_counts())  # returns unique classes in the species column (50)

# To better understand each attribute, start with univariate plots (single plot for each feature). Histograms are a
# type of bar chart that displays the counts or relative frequencies of values falling in different class intervals
# or ranges. There are more univariate summary plots including density plots and boxplots.
iris_df.hist()
plt.show()

# This gives us a much clearer idea of the distribution of the input variable, showing that both sepal length and
# sepal width have a normal (Gaussian) distribution. That is, the distribution has a beautiful symmetric bell shape.
# However, the length of petals is not normal. Its plot shows two modes, one peak happening near 0 and the other
# around 5. Less patterns were observed for the petal width.

# To see the interactions between attributes we use scatter plots. However, it's difficult to see if there's any
# grouping without any indication of the true species of the flower that a datapoint represents. Therefore,
# we define a color code for each species to differentiate species visually:

# build a dict mapping species to an integer code
iris_names = list(iris_df['species'].value_counts().keys())
print(iris_names)
indices = [i for i in range(len(iris_names))]
print(indices)

iris_names_dict = dict(zip(iris_names, indices))
for key, val in iris_names_dict.items():
    print('{}: {}'.format(key, val))

# build integer color code 0/1/2
colors = [iris_names_dict[species] for species in iris_df['species']]
# scatter plot
scatter_plot = plt.scatter(
    iris_df['sepal_length'],
    iris_df['sepal_width'],
    c=colors
)
plt.xlabel('Sepal length [cm]')
plt.ylabel('Sepal width [cm]')
plt.legend(
    handles=scatter_plot.legend_elements()[0],
    labels=iris_names
)
plt.show()
# Using sepal_length and sepal_width features, we can distinguish iris-setosa from others; separating iris-versicolor
# from iris-virginica is harder because of the overlap as seen by the green and yellow datapoints.

scatter_plot = plt.scatter(
    iris_df['petal_length'],
    iris_df['petal_width'],
    c=colors
)
plt.xlabel('Petal length [cm]')
plt.ylabel('Petal width [cm]')
plt.legend(
    handles=scatter_plot.legend_elements()[0],
    labels=iris_names
)
plt.show()

# Interestingly, the length and width of the petal are highly correlated, and these two features are very useful to
# identify various iris species. It is notable that the boundary between iris-versicolor and iris-virginica remains a
# bit fuzzy, indicating the difficulties for some classifiers. It is worth keeping in mind when training to decide
# which features we should use.

# To see scatter plots of all pairs of features, use pandas.plotting.scatter_matrix(). Besides the histograms of
# individual variables along the diagonal, it will show the scatter plots of all pairs of attributes to help spot
# structured relationships between features.
pd.plotting.scatter_matrix(iris_df)
plt.show()

iris_df.hist()
plt.show()

# K nearest neighbors
# K nearest neighbors (knn) is a supervised machine learning model that takes a data point,
# looks at its 'k' closest labeled data points, and assigns the label by a majority vote.

# Here we see that changing k could affect the output of the model. In knn, k is a hyperparameter. A hyperparameter
# in machine learning is a parameter whose value is set before the learning process begins. We will learn how to tune
# the hyperparameter later.

# K nearest neighbors can also be used for regression problems. The difference lies in prediction. Instead of a
# majority vote, knn for regression makes a prediction using the mean labels of the k closest data points.

# The value of k to use should be odd numbers. This is because, since the prediction of a new value is based on
# majority vote of nearest neighbors, we want to avoid a tie. Equally important to mention is, k=1 doesn't make good
# sense.

# Actually, the value of k must be optimized. Depending on data, small k value could be too restrictive,
# and then results in error (overfit). Similarly, large k value could too open/broad and results in error (underfit).

# Earlier we identified that the length and the width of the petals are the most useful features to separate the
# species; we then define the features and labels as follows:
X = iris_df[
    ['petal_length', 'petal_width']
].values
y = iris_df['species'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    shuffle=True,
    random_state=22
)

# Note that we specified the split was stratified by label (y). This is done to ensure that the distribution of
# labels remains the same for each target class label in both train and test sets.
# In classifications, stratified sampling is often chosen to ensure that the train and test sets have approximately
# the same percentage of samples of each target class as the complete set.
y_train_unique, y_train_counts = np.unique(y_train, return_counts=True)
y_test_unique, y_test_counts = np.unique(y_test, return_counts=True)
print(y_train_unique, y_train_counts)
print(y_test_unique, y_test_counts)

model_knn = KNeighborsClassifier(
    n_neighbors=5
)
model_knn.fit(X_train, y_train)
y_pred = model_knn.predict(X_test)
print(y_pred[:5], y_test[:5])

y_proba = model_knn.predict_proba(X_test)
print(y_proba[20:22])

# In classification the most straightforward metric is accuracy.
# It calculates the proportion of data points whose predicted labels exactly match the observed labels.
# Reminder:
# Accuracy = good predictions / total number of predictions
# ratio of correctly predicted observation to the total observations

# Precision = true positives / (true positives + false positives)
# Precision is the ratio of correctly predicted positive observations to the total predicted positive observations.
# Among all passengers that were labeled as survived, how many actually survived?
# High precision relates to the low false positive rate.

# Recall (Sensitivity) = true positives / (true positives + false negatives)
# ratio of correctly predicted positive observations to the all positive observations in actual class
# Among all the passengers that truly survived, how many did we label positively?

# F1 Score = 2 * (precision * recall) / (precision + recall)
# F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and
# false negatives into account. Intuitively it is not as easy to understand as accuracy, but F1 is usually more
# useful than accuracy, especially if you have an uneven class distribution. Accuracy works best if false positives
# and false negatives have similar cost. If the cost of false positives and false negatives are very different,
# it’s better to look at both Precision and Recall.

accuracy = accuracy_score(
    y_true=y_test,
    y_pred=y_pred
)
precision, recall, fscore, support = precision_recall_fscore_support(
    y_true=y_test,
    y_pred=y_pred
)
precision = np.mean(precision)
recall = np.mean(recall)
fscore = np.mean(fscore)
print('Accuracy of KNN Classifier with k=5:', accuracy)
print('Precision of KNN Classifier with k=5:', precision)
print('Recall of KNN Classifier with k=5:', recall)
print('F1 Score of KNN Classifier with k=5:', fscore)

# Classification accuracy alone can be misleading if there is an unequal number of observations in each class or if
# there are more than two classes in the dataset. Calculating a confusion matrix will provide a better idea of what
# the classification is getting right and what types of errors it is making.

# What is a confusion matrix? It is a summary of the counts of correct and incorrect predictions, broken down by each
# class.
confusion_matrix = confusion_matrix(
    y_true=y_test,
    y_pred=y_pred,
    labels=iris_names
)
print(confusion_matrix)

# A confusion matrix is a table that is often used to describe the performance of a classification model
# (or "classifier") on a set of test data for which the true values are known.
# Confusion Matrix confirmed our previous observation during exploratory data analysis showed, that there was some
# overlap between the two species on the scatter plot and it is more difficult to distinguish iris-versicolor
# from iris-virginica than identifying iris-setosa.
plot_confusion_matrix(
    model_knn,
    X_test,
    y_test,
    cmap=plt.cm.Blues
)
plt.show()

# Quick exercise regarding the confusion matrix:
# y_true = [dog, cat, cat, dog, dog]
# y_pred = [dog, cat, cat, cat, dog]
# cm.labels = [cat, dog]
# cm = [[2, 0]
#       [1 , 2]]
# we have 0 wrong predictions for cat, and 1 wrong prediction for dog


# Previously we made train-test split before fitting the model so that we can report the model performance on the
# test data. This is a simple kind of cross validation technique, also known as the holdout method. However,
# the split is random, as a result, model performance can be sensitive to how the data is split. To overcome this,
# we introduce k-fold cross validation.

# In k fold cross validation, the data is divided into k subsets. Then the holdout method is repeated k times,
# such that each time, one of the k subsets is used as the test set and the other k-1 subsets are combined to train
# the model. Then the accuracy is averaged over k trials to provide total effectiveness of the model. In this way,
# all data points are used; and there are more metrics so we don’t rely on one test data for model performance
# evaluation.

# The simplest way to use k-fold cross-validation in scikit-learn is to call the cross_val_score function on the
# model and the dataset (from sklearn.model_selection import cross_val_score

# create new KNN Classifier with k=3
new_knn = KNeighborsClassifier(
    n_neighbors=3
)
# fit the model using cross_val_score with 5 cross validations cv=5
# cv=5 means that we have 5 cross validations, so the dataset is divided into 5 equal data chunks
# 100% of dataset / 5 = each data subset contains 20% of whole dataset
cv_scores = cross_val_score(
    new_knn,
    X, y,
    cv=5
)
print(cv_scores)
new_accuracy = np.mean(cv_scores)
print('New Accuracy for KNN with k=3 using Cross Validation cv=5:', new_accuracy)

# As a general rule, 5-fold or 10-fold cross validation is preferred; but there is no formal rule. As k (cv) gets
# larger, the difference in size between the training set and the resampling subsets gets smaller. As this difference
# decreases, the bias of the technique becomes smaller.

# If the sample size of the dataset is large and the target class is symmetrically distributed, train_test_split
# could be fine. If the dataset is few in sample size, cross validation becomes more acceptable.
# Nonetheless, both can be used together especially if the sample size is large enough. Split the data, and use CV to
# evaluate performance on the training set. Then optimize the model by tuning hyperparameter, and after best model is
# selected do external validation with the split out (hold-out) test set.

# When we built our first knn model, we set the hyperparameter k to 5, and then to 3 later in k-fold cross
# validation; random choices really. What is the best k? Finding the optimal k is called tuning the hyperparameter. A
# handy tool is grid search. In scikit-learn, we use GridSearchCV, which trains our model multiple times on a range
# of values specified with the param_grid parameter and computes cross validation score, so that we can check which
# of our values for the tested hyperparameter performed the best.

# create new knn
default_knn = KNeighborsClassifier()
# create a grid, that is a dictionary containing a hyperparameter name as key,
# and respective values (single value or list of values) that we want to check the results for
params = [i for i in range(3, 10) if i % 2 != 0]
param_grid = {
    'n_neighbors': params
}
# use GridSearchCV to test all values for n_neighbors hyperparameter
knn_gscv = GridSearchCV(
    default_knn,
    param_grid=param_grid,
    cv=5
)
# fit the model using feature matrix X and target array y
knn_gscv.fit(X, y)
# check the best hyperparameters for our model, the best_params_ attribute will return the dictionary
print('Best hyperparameters for the KNN model:', knn_gscv.best_params_)
print('Best score for the KNN model with most optimal hyperparameters:', knn_gscv.best_score_)

# Now we can build the final model
knn_final = KNeighborsClassifier(
    n_neighbors=knn_gscv.best_params_['n_neighbors']
)
knn_final.fit(X, y)
y_pred_final = knn_final.predict(X)
print('Score for the final KNN model:', knn_final.score(X, y))

# The techniques of k-fold cross validation and tuning parameters with grid search is applicable to both
# classification and regression problems.

# ####Label prediction with new data#### Now we are ready to deploy the model 'knn_final'. We take some measurements
# of an iris and record that the length and width of its sepal are 5.84 cm and 3.06 cm, respectively, and the length
# and width of its petal are 3.76 cm and 1.20 cm, respectively.
# How do we make a prediction using the built model?

# The model was trained on feature matrix containing the petal_length and petal_width columns
new_data = np.array(
    [3.76, 1.20]
)
# remember that predict method takes a numpy 2darray or pandas DataFrame as input parameter!
# and because y is a 1darray, reshape(1, -1) for the input data to be a single row array, will be a good choice
y_pred_for_new_data = knn_final.predict(new_data.reshape(1, -1))
print('Prediction for the new data:', y_pred_for_new_data)
new_data = np.array([
    [3.76, 1.20],
    [5.25, 1.20],
    [1.58, 1.20]
])
y_pred_for_new_data = knn_final.predict(new_data)
print('Prediction for the new data:\n', y_pred_for_new_data)
y_proba_for_new_data = knn_final.predict_proba(new_data)
print('Probability for the new data:\n', y_proba_for_new_data)

# Finally, I found that it breaks ties based on 'lexicographic order of class names'. In depth: predict_proba method
# output is actually dependent on class labels lexicographic sort. Class labels are lexicographically sorted first
# and then class is picked using stats.mode (to find the majority of neighbours). numpy stats.mode breaks ties giving
# preference to lower index. This means, when a tie in this dataset it prefers the first(lower column index) flower
# over the other for prediction. Since lexicographically, iris-setosa < iris-versicolor < iris-virginica [0.5, 0.5,
# 0]  picks iris-setosa [0.2, 0.4, 0.4] picks iris-versicolor but if we simply renamed iris-setosa to iris-versicolor
# and iris-versicolor to iris-setosa in our y dataset, it would STILL pick as above due to the reason stated at the
# start: class labels are sorted always (predictor labels are not strictly mapped to their feature vectors) and are
# chosen according to stats.mode calculation.

# Excercise as a reminder:
# Which of the following are categorical data?
# 1) The breed of a dog, e.g., shepherd, terrier, chihuahua
# 2) You custom service experience, e.g., very poor, poor, neutral, good, very good
# 3) Housing prices in Los Angeles, California
# 4) Housing prices in Los Angeles, California
# Answer is 1, 2, 4,
# because categorical data is a type of data which may be divided into groups (classes with corresponding labels)

# import numpy as np
# from sklearn.metrics import confusion_matrix
# y_true = np.array(['cat', 'dog', 'dog', 'cat', 'fish', 'dog', 'fish'])
# y_pred = np.array(['cat', 'cat', 'cat', 'cat', 'fish', 'dog', 'fish'])
# confusion_matrix(
#                  y_true,
#                  y_pred,
#                  labels=['cat', 'dog', 'fish']
# )
# >>> [[2, 0, 0]
#      [2, 1, 0]
#      [0, 0, 2]]
# Answer cat as cat (2), dog as dog (1), dog as cat (2), fish as fish (2)
