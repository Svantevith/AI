# Gini Impurity
# the closer to 0 the more pure
# the closer to 0.5 the more impure
# gini = n * p * (1-p)
# where n is number of clases, in our case it is 2: Survived and Died, p is percent of survived


# Entropy
# the closer to 0 the more pure
# the closer to 1 the more impure
# the graph looks very similiarly to Gini but is a little thicker/wider
# entropy =  -(p*log(p) + (1-p)*log(1-p)
# 0 means completely pure, 100% of samples belong to 1 class
# 1 means completely impure, 50% of samples belong to one class and other 50% to the second

# Information Gain
# The higher the better
# Information Gain = H(Set) - H(A) * A/Set - H(B) * B/Set
# where H is either Gini or Entropy, A and B are left and right sets respectively (splitted by condition)

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')

# remember that we need to have only numeric data in our feature matrix (numpy array) for further computations
df['Male'] = df['Sex'] == 'male'

X = df[
    ['Pclass', 'Male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']
].values

y = df['Survived'].values

# set the random_state (seed) to an arbitrary value to always have the same random dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=22)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# remember that predict() method takes in 2D numpy array as parameter and returns a 1D array!
print(
    'Survived\n' if model.predict([X_test[0]]) == 1 else 'Not Survived\n'
)

# The process of training & predicting is the same as with the use of LogisticRegression model
# Let's compare the DecisionTreeClassifier and LogisticRegression

kf = KFold(n_splits=5, shuffle=True, random_state=10)

dt_accuracy_scores = []
dt_precision_scores = []
dt_recall_scores = []
dt_roc_auc_scores = []

lr_accuracy_scores = []
lr_precision_scores = []
lr_recall_scores = []
lr_roc_auc_scores = []

for train_indices, test_indices in kf.split(X):
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    # by default the criterion parameter is set to 'gini', we can change it to 'entropy'
    dt = DecisionTreeClassifier(criterion='entropy')
    dt.fit(X_train, y_train)
    dt_y_pred = dt.predict(X_test)

    dt_accuracy_scores.append(
        accuracy_score(y_test, dt_y_pred)
    )
    dt_precision_scores.append(
        precision_score(y_test, dt_y_pred)
    )
    dt_recall_scores.append(
        precision_score(y_test, dt_y_pred)
    )
    dt_roc_auc_scores.append(
        roc_auc_score(y_test, dt_y_pred)
    )

    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    lr_y_pred = lr.predict(X_test)

    lr_accuracy_scores.append(
        accuracy_score(y_test, lr_y_pred)
    )
    lr_precision_scores.append(
        precision_score(y_test, lr_y_pred)
    )
    lr_recall_scores.append(
        precision_score(y_test, lr_y_pred)
    )
    lr_roc_auc_scores.append(
        roc_auc_score(y_test, lr_y_pred)
    )

dt_accuracy = np.mean(dt_accuracy_scores)
dt_precision = np.mean(dt_accuracy_scores)
dt_recall = np.mean(dt_recall_scores)
dt_roc_auc = np.mean(dt_roc_auc_scores)

lr_accuracy = np.mean(lr_accuracy_scores)
lr_precision = np.mean(lr_accuracy_scores)
lr_recall = np.mean(lr_recall_scores)
lr_roc_auc = np.mean(lr_roc_auc_scores)

print('Decision Tree Classifier')
print('Accuracy: ', dt_accuracy)
print('Precision: ', dt_precision)
print('Recall: ', dt_recall)
print('Roc Auc Score: ', dt_roc_auc, '\n')

print('Logistic Regression Classifier')
print('Accuracy: ', lr_accuracy)
print('Precision: ', lr_precision)
print('Recall: ', lr_recall)
print('Roc Auc Score: ', lr_roc_auc, '\n')

# we can export our Decition Tree diagram block as .png file

# import graphviz as gv
# from IPython.display import Image
# from sklearn.tree import export_graphviz

# feature_names = ['Pclass', 'Male']
# X = df[feature_names].values
# y = df['Survived'].values

# dt_model = DecisionTreeClassifier(criterion='entropy')
# dt_model.fit(X, y)

# dot_file = export_graphviz(dt_model, #feature_names=feature_names)
# graph = gv.Source(dot_file)
# cleanup=True means that we do not save any extra files except the png image
# graph.render(filename='tree', format='png', cleanup=True)

# Reduce Overfitting in Decision Tree We use so called tree pruning in order to reduce the Overfitting, by shrinking
# the size of the tree We can define two types of pruning: pre-pruning relies on stopping building the tree before it
# becomes too big - Max Depth: grow the tree only up to certain height. If max depth/height is 3, there will be at
# most 3 splits for each data point - Leaf Size: do not split a node if the number of samples in that node is under
# the minimum threshold - Limit the Leaf Nodes - limit the total number of leave nodes in the tree in post-pruning we
# firstly build the tree, then after specific review we erase the desired leaves to make the tree smaller

# new model:
dt_model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=3,
    min_samples_leaf=2,
    max_leaf_nodes=10
)

# Use the cross validation to obtain several results and choose the best values for pre-pruning parameters

# we will have n-models
# where n = len(crit)*len(depth)*len(samples)*len(leaves)
# = 2*3*2*4 = 48
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 10],
    'min_samples_leaf': [1, 3],
    'max_leaf_nodes': [10, 20, 35, 50]
}

dt_model = DecisionTreeClassifier()
gs = GridSearchCV(
    dt_model,
    param_grid,
    scoring='f1',
    cv=5
)
# train the GridSearchCV on whole X, y sets since it uses Cross Validation and provides the very important attribute
# .best_params_ ! Because of randomness of the data in folds (chunks) the results for different models may slightly
# differ, however if their performance is very similiar, always choose the least complex one
gs.fit(X, y)
print('Best parameters: ', gs.best_params_)

# Decision Trees are very prone to overfitting, even for underfitting because after many splits it may occur that lot
# of leaves have single samoles and hence still are very powerful causing overfitting. That's why it is very
# important to tune the Decision Tree. The biggest adventage of Decision Tree is that it is computationally very
# expensice to build (train), but comoutationally very inexpensive to predict. Moreover they are easy to understand
# for non-technical users.
