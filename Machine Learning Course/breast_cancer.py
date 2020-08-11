from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.linear_model import LogisticRegression

# create a dataset object using a load_breast_cancer() constructor
cancer_dataset = load_breast_cancer()

# this scikit-learn dataset is similiar to pythons dictionary
print(cancer_dataset.keys())

# 'DESCR' ia a brief description
# We have 569 datapoints
# For each datapoint, there is total of 30 features:
# 10 measurements of the breast mass (radius, texture, perimeter, etc.) with 3 computed values: mean, std and worst
# There is also one target, which is either Malignant (cancerous) or Benign (not cancerous)
print(cancer_dataset['DESCR'], '\n')

# (569, 30) 569 rows x 30 columns
# 569 patients with 10 measurements * 3 values, total of 30 features
print(cancer_dataset['data'].shape)

# feature_names show all of the feature names (column names for our Pandas DataFrame)
print(cancer_dataset['feature_names'])

# target is 1-D array with 569 boolean values (569 rows, 0 columns)
print(cancer_dataset['target'])
print(cancer_dataset['target'].shape)

# 0 is malignant (cancerous), 1 is benign (not cancerous)
print(cancer_dataset['target_names'])

# create the Pandas DataFrame
# rows are data, columns are feature_names
breast_df = pd.DataFrame(data=cancer_dataset['data'], columns=cancer_dataset['feature_names'])
print(breast_df.head())

# now add the target to our Pandas DataFrame
breast_df['target'] = cancer_dataset['target']

# build a feature matrix X
X = breast_df[
    cancer_dataset['feature_names']
].values

# set the target matrix
y = breast_df['target'].values

# build the model using sklearn linear model LogisticRegression using the default solver, there is a computation
# error, that's why we need to use a different solver='liblinear' solvers 'sag' and 'saga' are used for very large
# datasets with large sets of features, while there the liblinear would work bettee, because it contains large
# dataset, but a small set of features
model = LogisticRegression(solver='liblinear')
model.fit(X, y)

# make predictions
y_pred = model.predict(X[:5])
y_true = cancer_dataset['target'][:5]

print('\n', y_pred, '\n', y_true, '\n')

score = model.score(X, y)
print('Score: %.2f' % score, '%')
