from sklearn.datasets import load_diabetes
from sklearn.linear_model import LogisticRegression, Ridge
import pandas as pd

diabetes_dataset = load_diabetes()
print('Keys: \n', diabetes_dataset.keys(), '\n')
print('Brief description: \n', diabetes_dataset['DESCR'], '\n')
print('Data (values): \n', diabetes_dataset['data'], '\n')
print('Feature names (keys): \n', diabetes_dataset['feature_names'], '\n')
print('Target: \n', diabetes_dataset['target'], '\n')

diabetes_df = pd.DataFrame(
    data=diabetes_dataset['data'],
    columns=diabetes_dataset['feature_names']
)

print(diabetes_df.head())
diabetes_df['target'] = diabetes_dataset['target']

# values attribute converts the matrices into numpy arrays
X = diabetes_df[
    diabetes_dataset['feature_names']
].values

y = diabetes_df['target'].values

# model = LogisticRegression(solver='liblinear')
model = Ridge(alpha=0)
model.fit(X, y)
score = model.score(X, y) * 100
print('Score: %.2f' % score, '%')
