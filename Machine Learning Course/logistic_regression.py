import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, \
    precision_recall_fscore_support, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, KFold

pd.options.display.max_columns = 6

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')

print('Pandas DataFrame: \n', df.head(10), '\n')
print('Description for the DataFrame: \n', df.describe(), '\n')

# 1) Make all columns numerical
# Instead of 'male' & 'female' strings, store boolean values which are readble by numpy
df['Male'] = (df['Sex'] == 'male')

# 2) Create a simplified features numpy array for the scatter plot using values attribute
X = df[
    ['Fare', 'Age']
].values

# 3) Set a target numpy array using values attribute
y = df['Survived'].values

print('Features: \n', X, '\n')
print('Target: \n', y, '\n')

# 4) Build LogisticRegression model
model = LogisticRegression()
# fit method takes two arguments: 2-D numpy array containing features and 1-D numpy array containing target values
model.fit(X, y)
# coefficients a, b are obtained from X, in case of 2 features in the X matrix
print(model.coef_)
# coefficient c is obtained from y
print(model.intercept_)

# model.coef_ and model.intercept_ return 1-D or 2-D numpy array in case of single or multiple keys
A = model.coef_[0][0]
B = model.coef_[0][1]
C = model.intercept_[0]

# Plot adjusted line over scattered values
y0 = np.linspace(0, 80, 100)
# Ax + By = C => x = (C - By) / A
x0 = (C - B * y0) / A
plt.plot(x0, y0, '-b', label='')
plt.xlabel('Fare')
plt.ylabel('Age')
plt.scatter(
    df['Fare'], df['Age'], c=df['Survived']
)
plt.grid()
plt.show()

# 5) Make predictions
# prepare a full features numpy array using values attribute
# 2-D feature numpy array is a numpy array containing 1-D numpy arrays for each client
X = df[
    ['Pclass', 'Male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']
].values
# prepare a target numpy array
y = df['Survived'].values

model = LogisticRegression()
model.fit(X, y)
model.predict(X)

client_0 = X[0]
print('\nFirst client: \n', client_0, '\n')

# A single row of 2-D np.array as X, is 1-D np.array, model.predict() takes only 2-D np.arrays as arguments - X[0] as 1-D row is converted into 2-D numpy array
print('Prediction for first client: \n',
      model.predict([X[0]])
      )

# Multiple rows of 2-D np.array like X[:5] is a 2-D np.array. Only a single row is treated as 1-D np.array, multiple are creating the 2-D np.array

y_pred = model.predict(X[:5])
y_true = y[:5]

print(y_pred)
print(y_true)

y_pred = model.predict(X)
y_true = y

# To find the number of correctly predicted clients use the numpy method sum(), which will sum up all of the cases satisfying the condition (y_pred == y_true)
correct = (y_pred == y_true).sum()
print('\nCorrectly predicted: \n', correct)

# total number of passengers stored in 1-D numpy! array is y.shape[0]
# np.array.shape[0] returns size of the array
correct_perc = correct / y.shape[0]
print(correct_perc, '%')

score = model.score(X, y)
print('Score: ', score, '%')

cm = confusion_matrix(
    y_true=df['Survived'],
    y_pred=model.predict(X),
    labels=[0, 1])

print('Confusion matrix: \n', cm)

# Regular confusion matrix:
# [[TP FP
#   FN TN]]

# Sklearn confusion matrix:
# [[TN FP
#   FN TP]]
TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TP = cm[1][1]

# Accuracy is the % of predictions that are correct
accuracy = (y_pred == y_true).sum() / y_true.shape

# Precision is the measurement how precise our model is with its positive predictions

precision = TP / (TP + FP)

# Recall is the measurement how many of the positive predictions our model can recall

recall = TP / (TP + FN)

# F1 Score is the harmonic mean of precision and recall

f_score = 2 * (precision * recall) / (precision + recall)

# But we will use the sklearn.metrics
# these methods take y_true and y_pred as parameters

accuracy = accuracy_score(y_true, y_pred)
print('Accuracy: \n{}'.format(accuracy))
precision = precision_score(y_true, y_pred)
print('Precision: \n{}'.format(precision))
recall = recall_score(y_true, y_pred)
print('Recall: \n{}'.format(recall))
f_score = f1_score(y_true, y_pred)
print('F1 Score: \n{}'.format(f_score))

# Now we will divide our dataset into the training and test sets
# by default the train_size=75, so it is train data : test data = 75 : 25 %
# without using the random_state (seed), we would always use different results. Setting the random_state (seed) to some arbitrary value will make you sure, that you will always get the same random distribution of data

print('\n##### Dataset division into training and testing sets #####\n')

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=80, random_state=37)

# After spliting the feature matrix X.shape (887, 7) into the training and testing set, by default we will obtain two 2D arrays (0.75*887, 7) and (0.25*887, 7)
# After splitting the target array y.shape (887,) into the train and test batches, we get two arrays 1D (887*0.75,) and (887*0.25,)

print('X size: ', X.shape)
print('X_train size: ', X_train.shape)
print('X_test size: ', X_test.shape)
print('y size: ', y.shape)
print('y_train size: ', y_train.shape)

# Then we build and train our model on the training set

model = LogisticRegression()
model.fit(X_train, y_train)

# The predictions are evaluated using the features test set

y_pred = model.predict(X_test)

# Accuracy, precision, recall and f1 scores are obtained using the test and predictions target datasets

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: \n{}'.format(accuracy))
precision = precision_score(y_test, y_pred)
print('Precision: \n{}'.format(precision))
recall = recall_score(y_test, y_pred)
print('Recall: \n{}'.format(recall))
f_score = f1_score(y_test, y_pred)
print('F1 Score: \n{}'.format(f_score))

# ROC Receiving Operating Characteristic is a graph showing all possible models and their performance

# we can also use the precision_recall_fscore_support from sklearn.metrics
# in order to obtain precision, recall and f1 scores
# we also get the Sensitivity and Specificity

# Sensitivity is the true positive rate (Recall)
# Sensitivity = TP / TP + FN
# Specificity is the true negative rate
# Specificity = TN / TN + FP

precision, recall, f_score, support = precision_recall_fscore_support(y_test, y_pred)

# Recall is an array consisting of 2 values:
# 1st value is the recall for true negative rate (Specificity)
# 2nd value is the recall for the true positive raye (Sensitivity)
# Support is the number of occurences of each label

specificity = recall[0]
print('True negative rate (Specificity): ', specificity)
sensitivity = recall[1]
print('True positive rate (Sensitivity): ', sensitivity)
print('Support: ', support)

# Adjusting the Logistic Regression Thresholds
# by default it is 0.5

# let's the the probsbility for each label
# remember that for predicting we use the testing features dataset X_test for evaluation
# model.predict_proba(X_test) returns 2D array constisting of the probability for negative cases (one-hot encoded as 0) [0] and for positive cases (one-hot encoded as 1)
y_pred_proba = model.predict_proba(X_test)
print('Probability array for first 5 people: ', y_pred_proba[:5, ])

# we set the threshold to 0.75 for our model to classify these predictions as positive starting not from 0.5 as by default (LogisticRegression), but from the new threshold of 0.75. It will make the model for precise in case of predicting the positive cases. The sensitivity/recall (true positive rate) will increase

print(
    '\n##### Dataset division into training and testing sets while using the higher threshold 0.75 for predicting the positive cases #####\n')

# all of the people, only classified as survivors (2nd column of the probability array) having the probability of above 0.75
y_pred = model.predict_proba(X_test)[:, 1] > 0.75

precision, recall, f_score, support = precision_recall_fscore_support(y_test, y_pred)

print('Precision: ', precision)
print('Recall: ', recall)
print('F1 Score: ', f_score)

specificity = recall[0]
print('True negative rate (Specificity): ', specificity)
sensitivity = recall[1]
print('True positive rate (Sensitivity): ', sensitivity)

# Plotting the ROC Curve showing the sensitivity vs specificity ratio
# every predicted probability is the threshold
# roc_curve returns array of false positives FP, true positives TP and the thresholds
# the false positive rate is 1-specificity (x-axis)
# the true positive rate is sensitivity

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_test)

# the ROC curve will show us the best threshold to build our model

FP_rate, TP_rate, thresholds = roc_curve(y_test, y_pred_proba[:, 1])

# The more to the left, the higher the specificity (true negative rate)
# remember that ROC curve shows False Positive rate (1-Specificity) on the x-axis, so the closer 0 on x-axis, the higher specificity
# The more up, the higher the sensitivity (true positive rate)
# moreover the ROC curve should never fall below the diagonal line, it would mean that our kodel predicts worse than a random model!

plt.plot(FP_rate, TP_rate)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid()
plt.show()

# As an example I have three points:
# A(0.1, 0.65) FP=0.1(TN=0.9), TP=0.65
# B(0.4, 0.75) FP=0.4(TN=0.6), TP=0.75
# C(0.7, 0.9) FP=0.7(TN=0.3), TP=0.9

# model A should be chosen if it is more important to have all of the positive predictions correct, meaning that the false positive rate is very small - it is better to have less uncorrectly predicted positive cases and the model should have high Specificity
# model C should be chosen if it is more important to predict as many possible positive cases as possible, while having many uncorrectly predicted positive cases is not a problem, a model with higher Sensitivity (True positive rate) is a right choice
# model B is the balanced choice maintaining the golden mean of specificity and sensitivity

# Basically the best possible model would be located mostly on the left upper path of the graph
# The higher Area Under Curve AUC the better performance of the model

model_1 = LogisticRegression()
model_1.fit(X_train, y_train)
y_pred_1 = model_1.predict_proba(X_test)[:, 1]
auc_1 = roc_auc_score(y_test, y_pred_1)

# Model 2 has shrinked number of features - the features dataset for second model contains only first 3 features of Model 1
model_2 = LogisticRegression()
model_2.fit(X_train[:, :3], y_train)
y_pred_2 = model_2.predict_proba(X_test[:, :3])[:, 1]
auc_2 = roc_auc_score(y_test, y_pred_2)

print('AUC Score for Model 1', auc_1)
print('AUC Score for Model 2', auc_2)

# Instead of splitting our dataset a single time, we are going to split the dataset into training and test set multiple times using the K-Fold Cross Validation

# During each evaluation we get different results for each test set
# Instead of taking a chunk from our dataset and making it a test set, let's split the data into chunks. Each of the chunks are going to serve as test sets now.
# Let's assume a dataset of 200 datapoints:
# 1) t t t t T
# 2) t t t T t
# 3) t t T t t
# 4) t T t t t
# 5) T t t t t
# we can notice that still we have 20% of test and 80% of training samples
# where t is training set and T is test set
# each datapoint is in exactly 1 test set
# now we have 5 different datasets containing different training and test sets
# we build models having own training sets and we evaluate it using associated test sets
# then we compare results and on this basis we create a new model using all of the data
# this process is called k-fold cross validation, where k is number of chunks we divide our dataset into
# our goal in cross validation is to obtain accurate measures of our metrics
# averging of all possible values will eliminate the impact of which database our test set lands in
# building extra models to properly calculate different results will help to make correct decisions, because of properly evaluated extra metrics
# usually training the smaller datasets using the k-fold validation gives higher scores, due to larger number of evalutation and thus building more appropriate final model

# for large datasets only single split is used for simplicity

print('\n##### K-Fold Cross Validation ####\n')

kf = KFold(n_splits=5, shuffle=True)
for train, test in kf.split(X[:10]):
    print(train, test)

print('\nScores for each of the k = 5 folds: ')
scores = []
for chunk in list(kf.split(X)):
    train_indices, test_indices = chunk

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    model = LogisticRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    scores.append(score)
    print(score)

mean_score = np.mean(scores)
print('Mean score: ', mean_score)

# Even though each chunk is different and the results are different each time we compute them using k-fold cross validation, the final mean score should be very similiar each epoch
# any metrics found during the cross validation are valid for the final model built on full dataset

final_model = LogisticRegression()
final_model.fit(X, y)

# Normally in any ML workflow, even if two features are highly relevant, it may not be a good idea to add both of them to the training set if they are highly correlated. In that case, adding both features would not only increase the model complexity (increasing the possibility of overfitting),but also would not add significant information, due to the correlation between the features.

# Therefore, the best way is to choose a minimal number of features that impact the prediction of the model. Choosing more features than desired is known as the curse of dimensionality (the unintuitive and sparse properties of data in high dimensions) in Machine Learning and should be avoided.

print('\n##### Choosing right model #####\n')


def scores_descr(X, y, kf, characteristics):
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    for chunk in list(kf.split(X)):
        train_indices, test_indices = chunk
        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]

        model = LogisticRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict_proba(X_test)[:, 1] > 0.65

        accuracy_scores.append(
            accuracy_score(y_test, y_pred)
        )
        precision_scores.append(
            precision_score(y_test, y_pred)
        )
        recall_scores.append(
            recall_score(y_test, y_pred)
        )
        f1_scores.append(
            f1_score(y_test, y_pred)
        )

    characteristics.append(
        {
            'accuracy': np.mean(accuracy_scores),
            'precision': np.mean(precision_scores),
            'recall': np.mean(recall_scores),
            'f1 score': np.mean(f1_scores),
        }
    )


X1 = df[
    ['Pclass', 'Male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']
].values

X2 = df[
    ['Pclass', 'Male', 'Age']
].values

X3 = df[
    ['Fare', 'Age']
].values

y = df['Survived'].values

kf = KFold(n_splits=5, shuffle=True)

characteristics = []
features_datasets = (X1, X2, X3)

scores_descr(X1, y, kf, characteristics)
print('Model with all features')
for key, value in characteristics[0].items():
    print('{}: {}'.format(key, value))

scores_descr(X2, y, kf, characteristics)
print("\nModel with 3 features ['Pclass', 'Male', 'Age']")
for key, value in characteristics[1].items():
    print('{}: {}'.format(key, value))

scores_descr(X3, y, kf, characteristics)
print("\nModel with 2 features ['Fare', 'Age']")
for key, value in characteristics[2].items():
    print('{}: {}'.format(key, value))