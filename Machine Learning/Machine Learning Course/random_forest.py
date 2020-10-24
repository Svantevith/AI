# Decision Trees are very susceptible to random idiosyncrasies in the training dataset. We say that Decision Trees
# have high variance since if you randomly change the training dataset, you may end up with a very different looking
# tree.

# One of the advantages of decision trees over a model like logistic regression is that they make no assumptions
# about how the data is structured. In logistic regression, we assume that we can draw a line to split the data.
# Sometimes our data just isn’t structured like that. A decision tree has the potential to get at the essence of the
# data no matter how it is structured.

# We will be learning about random forests in this module, which as you may guess from the name, is a model built
# with multiple trees. The goal of random forests is to take the advantages of decision trees while mitigating the
# variance issues.

# A random forest is an example of an ensemble because it uses multiple machine learning models to create a single
# model.

# A bootstrapped sample is a random sample of datapoints where we randomly select with replacement datapoints from
# our original dataset to create a dataset of the same size. Randomly selecting with replacement means that we can
# choose the same datapoint multiple times. This means that in a bootstrapped sample, some datapoints from the
# original dataset will appear multiple times and some will not appear at all.

# ABCDE - AABCD, ABBDD, BBBBA, CDDAA, ABCEE We use bootstrapping to mimic creating multiple samples.

# Bootstrap Aggregation (or Bagging) is a technique for reducing the variance in an individual model by creating an
# ensemble from multiple models built on bootstrapped samples.

# To make a prediction, we make a prediction with each of the 10 decision trees and then each decision tree gets a
# vote. The prediction with the most votes is the final prediction.

# When we bootstrap the training set, we're trying to wash out the variance of the decision tree. The average of
# several trees that have different training sets will create a model that more accurately gets at the essence of the
# data.

# With bagged decision trees, the trees may still be too similar to have fully created the ideal model. They are
# built on different resamples, but they all have access to the same features. Thus we will add some restrictions to
# the model when building each decision tree so the trees have more variation. We call this decorrelating the trees.

# A standard choice for the number of features to consider at each split is the square root of the number of
# features. So if we have 9 features, we will consider 3 of them at each node (randomly chosen).

# If we bag these decision trees, we get a random forest. Each decision tree within a random forest is probably worse
# than a standard decision tree. But when we average them we get a very strong model!

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from math import sqrt
import matplotlib.pyplot as plt

cancer_dataset = load_breast_cancer()
df = pd.DataFrame(
    data=cancer_dataset['data'],
    columns=cancer_dataset['feature_names']
)

if 'target' not in df.keys():
    df['target'] = cancer_dataset.target

X = df[cancer_dataset.feature_names].values
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=17)

rf = RandomForestClassifier(random_state=17)
rf.fit(X_train, y_train)
# default score is accuracy
print('Accuracy of default Random Forest Tree:', rf.score(X_test, y_test))

# Let's compare it to the Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
print('Accuracy of default Decision Tree:', dt.score(X_test, y_test))

# Let's tune these models
# Random Forest Tree usually performs very well on default settings

n_features = X.shape[1]
print(int(np.round(sqrt(n_features) / 2)))
rf_param_grid = {
    'max_features': [int(np.round(sqrt(n_features) / 2)),
                     int(np.round(sqrt(n_features))),
                     int(np.round(sqrt(n_features) * 2))],
    'n_estimators': [10, 25, 50, 75]
}

rf_gs = GridSearchCV(
    rf,
    rf_param_grid,
    # scoring='f1
    cv=5
)

rf_gs.fit(X, y)
print('Best parameters for Random Forest Tree:', rf_gs.best_params_)

best_max_features = rf_gs.best_params_['max_features']
print(best_max_features)
best_n_estimators = rf_gs.best_params_['n_estimators']
print(best_n_estimators)

dt_param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7, 9],
    'min_samples_leaf': [3, 5, 10, 12],
    'max_leaf_nodes': [5, 10, 15, 20]
}

dt_gs = GridSearchCV(
    dt,
    dt_param_grid,
    # scoring='f1',
    cv=5
)

dt_gs.fit(X, y)
print('Best parameters for Decision Tree:', dt_gs.best_params_)
best_criterion = dt_gs.best_params_['criterion']
best_max_depth = dt_gs.best_params_['max_depth']
best_max_leaf_nodes = dt_gs.best_params_['max_leaf_nodes']
best_min_samples_leaf = dt_gs.best_params_['min_samples_leaf']

# Now build a model on these parameters
tuned_rf = RandomForestClassifier(
    max_features=best_max_features,
    n_estimators=best_n_estimators
)
tuned_rf.fit(X_train, y_train)
print('Accuracy for tuned Random Forest:', tuned_rf.score(X_test, y_test))

tuned_dt = DecisionTreeClassifier(
    criterion=best_criterion,
    max_depth=best_max_depth,
    min_samples_leaf=best_min_samples_leaf,
    max_leaf_nodes=best_max_leaf_nodes
)
tuned_dt.fit(X_train, y_train)
print('Accuracy for tuned Decision Tree:', tuned_dt.score(X_test, y_test))

# With a parameter like the number of trees in a random forest, increasing the number of trees will never hurt
# performance. Increasing the number trees will increase performance until a point where it levels out. The more
# trees, however, the more complicated the algorithm. A more complicated algorithm is more resource intensive to use.
# Generally it is worth adding complexity to the model if it improves performance but we do not want to unnecessarily
# add complexity.

# We can use what is called an Elbow Graph to find the sweet spot. Elbow Graph is a model that optimizes performance
# without adding unnecessary complexity.

# number of trees in the forest
n_estimators = [i for i in range(1, 101)]
# number of features considered at each split
max_features = [i for i in range(1, int(np.round(sqrt(n_features) * 2)) + 1)]
param_grid = {
    # 'max_features': max_features,
    'n_estimators': n_estimators
}
gs = GridSearchCV(
    rf,
    param_grid,
    cv=5
)
gs.fit(X, y)
scores = gs.cv_results_['mean_test_score']
print(scores)

plt.plot(n_estimators, scores)
plt.xlabel('Number of estimators')
plt.ylabel('Accuracy score [default]')
plt.xlim(0, 100)
plt.ylim(0.9, 1)
plt.show()

# We should choose about 10 to be our number of estimators, because we want the minimum number of estimators that still
# yield maximum performance.
# We are looking for that yield point, which provides the highest score, while having the least n_estimators.


# Feature importance Random forests provide a straightforward method for feature selection: mean decrease impurity.
# Recall that a random forest consists of many decision trees, and that for each tree, the node is chosen to split
# the dataset based on maximum decrease in impurity, typically either Gini impurity or entropy in classification.
# Thus for a tree, it can be computed how much impurity each feature decreases in a tree. And then for a forest,
# the impurity decrease from each feature can be averaged. Consider this measure a metric of importance of each
# feature, we then can rank and select the features according to feature importance.

feature_importance = pd.Series(
    tuned_rf.feature_importances_,
    index=cancer_dataset.feature_names
).sort_values(ascending=False)

print(feature_importance.head(10))

# We can notice high importance of 'worst' features
worst_features = [feature for feature in cancer_dataset.feature_names if 'worst' in feature]
print(worst_features)

# Train the model on new feature matrix
X_worst = df[worst_features].values
y = df['target'].values
X_train, X_test, y_train, y_test = train_test_split(X_worst, y, random_state=17)

final_rf = tuned_rf
final_rf.fit(X_train, y_train)

print('Accuracy for the final Random Forest:', final_rf.score(X_test, y_test))

# Probably the biggest advantage of Random Forests is that they generally perform well without any tuning. They will
# also perform decently well on almost every dataset.

# A linear model, for example, cannot perform well on a dataset that cannot be split with a line. It is not possible
# to split the following dataset with a line without manipulating the features. However, a random forest will perform
# just fine on this dataset.

# When looking to get a benchmark for a new classification problem, it is common practice to start by building a
# Logistic Regression model and a Random Forest model as these two models both have potential to perform well without
# any tuning. This will give you values for your metrics to try to beat. Oftentimes it is almost impossible to do
# better than these benchmarks
