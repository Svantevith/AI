# Machine learning is a set of tools used to build models on data. Building models to understand data and make
# predictions is an important part of a data scientists' job.

# Both regression and classification problems are supervised learning problems.

# Regression is predicting a numerical value (for example, predicting what price a house will sell for) and
# classification is predicting what class something belongs to (for example, predicting if a borrower will default on
# their loan).

# Popular classification techniques:
# • Logistic Regression
# • Decision Trees
# • Random Forests
# • Neural Networks

# Scikit-learn, one of the best known machine learning libraries in python for machine learning, implements a large
# number of commonly used algorithms. Regardless of the type of algorithm, the syntax follows the same workflow:
# import > instantiate > fit > predict. Once the basic use and syntax of Scikit-learn is understood for one model,
# switching to a new algorithm is straightforward. Thus for the rest of the course we will be working with
# scikit-learn to build machine learning models in different use cases.

# ####LINEAR REGRESSION#### We start with linear regression, a simple supervised learning model. Linear regression
# fits a straight line to data, mathematically: y = m * x + b where b is the intercept and m is the slope,
# x is a feature or an input, whereas y is label or an output. Our job is to find m and b such that the errors are
# minimized. Mathematically, the distance between the fitted line and data points are calculated by residuals,
# indicated by the dashed black vertical line in the plot. So linear regression essentially is finding the line where
# it minimizes the sum of the squared residuals. Reminder: Variance = sum of squared residuals / total number of
# datapoints

# Linear regression models are popular because they can perform a fit quickly, and are easily interpreted. Predicting
# a continuous value with linear regression is a good starting point.

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

boston_dataset = load_boston()
boston_df = pd.DataFrame(
    data=boston_dataset.data,
    columns=boston_dataset.feature_names
)

# as we can see, there is no 'MEDV' column, which is our target feature_name
print('Current DataFrame columns: \n', boston_df.columns)

# we have to add the 'MEDV' column
boston_df['MEDV'] = boston_dataset.target

# It is useful for quickly testing if the DataFrame has the right type of data in it. To see the first few rows of a
# DataFrame, use .head(n), where you can specify n for the number of rows to be selected. If n is omitted,
# by default, it selects the first 5 rows. Often datasets are loaded from other file formats (e.g., csv, text),
# it is a good practice to check the first and last few rows of the dataframe and make sure the data is in a
# consistent format using head and tail, respectively.

# To check the first 5 rows, use boston.head(), for the ease of display, we select columns CHAS, RM, AGE, RAD, and MEDV:
print('\n',
      boston_df[
          ['CHAS', 'RM', 'AGE', 'RAD', 'MEDV']
      ].head(n=5)
      )

# Remember that we can get the Summary Statistics for certain rows and columns!

# Binary CHAS has a mean of 0.07, and its 3rd quartile is 0. This indicates that most of the values in CHAS are 0.
# The average number of rooms per dwelling ranges from 3.56 to 8.78, with a mean of 6.28 and a median of 6.21. The
# distribution of RM appears symmetric.

# Reminder: the closer the median is to mean, the more symmetric the specific dataset is
print('\nSummary Statistics for Boston Dataset:\n', boston_df.describe(include='all').round(2))

# Summary statistics provides a general idea of each feature and the target, but visualization reveals the
# information more clearly. It’s a good practice to visualize and inspect the distribution column by column. Here we
# look at CHAS and RM to verify our conclusions from the last part.

# CHAS only takes on two values, 0 and 1, with most of them 0’s. It is consistent with what the describe() reports;
# specifically, the third quartile of CHAS is 0.
chas_cnt = boston_df['CHAS'].value_counts()
plt.style.use('ggplot')
chas_cnt.plot(
    kind='bar'
)
plt.show()

# The distribution of RM appears normal and symmetric. The symmetry aligns with what we observed from the output of
# describe(), as the mean of RM 6.28 is close to its median 6.21.
boston_df.hist(
    column='RM',
    bins=20
)
plt.show()

# We can obtain similiar results using the pandas.DataFrame.plot() method
boston_df['RM'].plot(
    kind='hist',
    bins=20
)
plt.show()

# Informative data visualization not only reveals insights, but they are invaluable to communicate findings to
# stakeholders.

# Correlation Matrix To understand the relationship among features (columns), a correlation matrix is very useful in
# the exploratory data analysis. Correlation measures linear relationships between variables. We can construct a
# correlation matrix to show correlation coefficients between variables. It is symmetric where each element is a
# correlation coefficient ranging from -1 and 1. A value near 1 (resp. -1) indicates a strong positive (resp.
# negative) correlation between variables. We can create a correlation matrix using the "corr" function:
corr_matrix = boston_df.corr().round(2)

# The last row or column is used to identify features that are most correlated with the target MEDV (median value of
# owner-occupied homes in $1000’s). LSTAT (percentage of lower status of the population) is most negatively
# correlated with the target (-0.74) which means that as the percentage of lower status drops (the percentage of
# middle and high status population increases, the society becomes richer), the median house values increases; while
# RM (the average number of rooms per dwelling) is most positively correlated with MEDV (0.70) which means that the
# house value increases as the number of rooms increases.

# The parity (positive or negative) just indicates the type of relationship the features share. The closer to -1/+1
# to stronger the relationship between features is!

# Close to +1 --> as one variable increases, so does the other (direct relationship)

# Close to -1 --> as one variable increases, the other decreases (inverse relationship)

# Close to 0 --> you can't expect an increase in one variable to predict the other's behavior well (no/minimal
# relationship)
print('\nCorrelation matrix:\n', corr_matrix)

# ####Data Preparation#### Feature selection is used for several reasons, including simplification of models to make
# them easier to interpret, shorter training time, reducing overfitting, etc.

# In the previous lesson, we noticed that RM and MEDV are positively correlated. Recall that scatter plot is a useful
# tool to display the relationship between two features; let’s take a look at the scatter plot. using
# pandas.DataFrame.plot() we don't need to specify the labels for axes, x and y does it for us
boston_df.plot(
    kind='scatter',
    x='RM',
    y='MEDV'
)
plt.show()

# The price increases as the value of RM increases linearly. There are a few outliers that appear to be outside of
# the overall pattern. For example, one point on the center right corresponds to a house with almost 9 rooms but a
# median value slightly above $20K. Homes with similar values usually have around 6 rooms. In addition,
# the data seems to have a ceiling; that is the maximum median value is capped at 50.

# On the other hand prices tend to decrease with an increase in LSTAT; and the trend isn’t as linear.
boston_df.plot(
    kind='scatter',
    x='LSTAT',
    y='MEDV'
)
plt.show()

# Of the two features, RM appears a better choice for predicting MEDV (using Linear Regression). Thus we start with a
# univariate linear regression: MEDV = b + m * RM.

# In scikit-learn, models require a two-dimensional feature matrix (X, 2darray or a pandas DataFrame) and a
# one-dimensional target array (y).

# Here we define the feature matrix as the column RM in boston and assign it to X. Note the double brackets around
# 'RM' in the code below, it is to ensure the result remains a DataFrame, a 2-dimensional data structure:
X = boston_df[
    ['RM']
].values

y = boston_df['MEDV'].values

# Recall that the single bracket outputs a Pandas Series, while a double bracket outputs a Pandas DataFrame,
# and the model expects the feature matrix X to be a 2darray.

model_lr = LinearRegression()

# A good rule of thumb is to split data 70-30, that is, 70% of data is used for training and 30% for testing. We use
# train_test_split function inside scikit-learn’s module model_selection to split the data into two random subsets.
# Set random_state so that the results are reproducible.
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    shuffle=True,
                                                    random_state=42
                                                    )

# In short, fitting is equal to training. It fits the model to the training data and finds the coefficients specified
# in the linear regression model, i.e., intercept and slope. After it is trained, the model can be used to make
# predictions. Fitting is how well the machine learning model measures against the data upon which it was trained.
model_lr.fit(X_train, y_train)

# The linear regression model has been fitted, what it means is that both parameters, the intercept and the slope,
# have been learned.

# Our model y = mx + b => MEDV = RM * x + b

# The two parameters represent the intercept and slope of the line fit to the data. Our fitted model is MEDV = -30.57
# + 8.46 * RM. For one unit increase in RM, the median home price would go up by $8460.
m = model_lr.coef_.round(2)
b = model_lr.intercept_.round(2)

# Once the model is trained, supervised machine learning will evaluate test data based on previous predictions for
# the unseen data. We can make a prediction using the predict() method.

# When the average number of rooms per dwelling is 6.5, the model predicts a home value of $24,426.06
rm_sample = np.median(boston_df['RM'])
# .round()

# Note that the input to predict() method has to be 2-dimensional, either a 2darray or DataFrame will work in this case.
# we need to reshape our single rm value into 2darray
rm_sample_2d = np.array(
    rm_sample
).reshape(-1, 1)

pred_for_sample = model_lr.predict(
    rm_sample_2d
)
print('\nPrediction for the house with median RM value = {}\n{}'.format(rm_sample, pred_for_sample))

# This value is the same as we plug in the line b + m*x where b is the estimated intercept from the model,
# and m is the estimated slope.
lp_for_sample = m * rm_sample + b
print('\nPoint on the line for the house with median RM value = {}\n{}'.format(rm_sample, lp_for_sample))

# predict method returns a 1darray, however it takes a 2darray as input!
y_pred = model_lr.predict(X_test)
print('\nPrediction for first 5 houses:\n', y_pred[:5])

# How good is our prediction? We can examine model performance by visually comparing the fitted line and the true
# observations in the test set

# Scatter plot of the true values y_test based on X_test
plt.scatter(
    X_test,
    y_test,
    label='Test set (true values)'
)
# Line plot showing the predictions y_pred based on X_test
plt.plot(
    X_test,
    y_pred,
    label='Predictions (based on X_test)',
    linewidth=2
)
plt.legend(
    loc='upper left'
)
plt.xlabel('RM')
plt.ylabel('MEDV')
plt.title('Correlation between RM and MEDV')
plt.show()

# Some points sit on the line, but some are away from it. We can measure the distance between a point to the line
# along the vertical line, and this distance is referred to as residual or error. A residual is the difference
# between the observed value of the target and the predicted value. The closer the residual is to 0, the better job
# our model is doing.

# We can calculate a residual and represent it in a scatter plot.

# Residuals are scattered around the horizontal line, y = 0, with no particular pattern. This seemingly random
# distribution is a sign that the model is working. Ideally the residuals should be symmetrically and randomly spaced
# around the horizontal axis; if the residual plot shows some pattern, linear or nonlinear, that’s an indication that
# our model has room for improvement.
residuals = y_test - y_pred

# Plot scatter, showing the residual (difference between true and prediction value) based on X_test
plt.scatter(
    X_test,
    residuals,
)
# The further the residual point lays from the y=0 line, the higher the error and the further prediction from
# its true value is
plt.hlines(
    y=0,
    xmin=X_test.min(),
    xmax=X_test.max(),
    linestyle='--'
)
plt.xlabel('RM')
plt.ylabel('Residuals')
plt.xlim(4, 9),
plt.show()

# Residual plots can reveal bias from the model and statistical measures indicate goodness-of-fit.
# Residuals show the errors of predictions

# There is a problem with residuals - they can be both positive or negative, that's why taking the average
# is not an accurate metric (negative cancel out the negative values).
# In order to solve it, we can use the Mean Squared Error method (MSE),
# make square of each residual and take the mean of squares:
mse1 = (residuals ** 2).mean()
print("Mean Squared Error of residuals for target 'MEDV'", mse1)

# Just use the mean_squared_error method from sklearn.metrics
mse1 = mean_squared_error(
    y_true=y_test,
    y_pred=y_pred
)
print("Mean Squared Error", mse1)

# In general the smaller the Mean Squared Error MSE, the better, yet there is no absolute good or bad threshold.
# We can define it based on the dependent variable, in our case, 'MEDV' in the test set.
# y_test ranges fro 6.3 to 50 with variance 84.42
# MSE in comparison to 'MEDV' variance, is not bad.
medv_var = np.var(boston_df['MEDV'])
print("Variance of 'MEDV' column", medv_var)

# To make the scale of errors to be the same as the scale of targets, root mean squared error (RMSE) is often used.
# It is the square root of MSE.
rmse1 = np.sqrt(mse1)
print("Root Mean Squared Error", rmse1)

# Mean Absolute Error (MAE) is another common method of evaluating model error. You take the absolute value of the
# residuals, and then find the average.  This way negative errors can no long cancel out positive errors
mae1 = mean_absolute_error(
    y_true=y_test,
    y_pred=y_pred
)
print("Mean Absolute Error", mae1)

# Another common metric to evaluate the model performance is called R-squared; one can calculate it via model.score():
# model.score(X_test, Y_test)
# R-Squared is the proportion of total variation explained by the model.
# Here, around 46% of variability in the testing data is explained by our model.
r_squared1 = model_lr.score(X_test, y_test)
print('R-Squared', r_squared1)

# The total variation is calculated as the sum of squares of the difference between the response and
# the mean of response, in the example of testing data:
total_var = ((y_test - y_test.mean()) ** 2).sum()
# Whereas the variation that the model fails to capture is computed as the sum of squares of residuals:
var_fail = (residuals ** 2).sum()
# Then the proportion of total variation from the data is:
proportion_of_total_var = 1 - (var_fail / total_var)
print('Proportion of total variation', proportion_of_total_var)

# A perfect model explains all the variation in the data. Note R-squared is between 0 and 100%:
# 0% indicates that the model explains none of the variability of the response data around its mean
# while 100% indicates that the model explains all of it.

# Evaluating R-squared values in conjunction with residual plots quantifies model performance.

# We use this R-squared value as a way to measure how much variability of the data our model can reliably explain
# in its current state. Of course, it makes sense that our model can only explain about 60% of real data variability -
# in real life, you'd expect much more to go into home valuation than just how many rooms it has
# (though the size of the house clearly has the largest impact on price).


# ####MULTIVARIATE LINEAR REGRESSION#### Recall 'LSTAT' (% lower status in population) is most negatively correlated
# to the home price. We can add the feature and build a multivariate linear regression model where the home price
# depends on both RM and LSTAT linearly:
print('Correlation matrix', boston_df.corr().round(2))

# MEDV = m1 * RM + m2 * LSTAT + b To find intercept b0, and coefficients b1 and b2, all steps are the same except
# for the data preparation part, we are now dealing with two features:

X2 = boston_df[
    ['RM', 'LSTAT']
].values

y2 = boston_df[
    'MEDV'
].values

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2,
    y2,
    test_size=0.3,
    shuffle=True,
    random_state=22
)

model_lr2 = LinearRegression()
model_lr2.fit(X2_train, y2_train)

m1, m2 = model_lr2.coef_
b2 = model_lr2.intercept_
print('Slope for RM {}, slope for LSTAT {}, intercept {},'.format(m1, m2, b))

y_pred2 = model_lr2.predict(X2_test)
print('Prediction for the first house', y_pred2[0])
print("Prediction for the first house using line equation",
      m1 * X2_test[0, 0] + m2 * X2_test[0, 1] + b2
      )

# The extension from univariate (single column in feature matrix X) to multivariate
# (multiple columns in feature matrix X) linear regression is straightforward in scikit-learn.
# The model instantiation, fitting, and predictions are identical, the only difference being the data preparation.

# As the name implies, multivariate regression is a technique that estimates a single regression model with more than
# one outcome variable. When there is more than one predictor variable in a multivariate regression model,
# the model is a multivariate multiple linear regression model.

# Which model is better? An easy metric for linear regression is the mean squared error (MSE) on the testing data.
# Better models have lower MSEs.
print('MSE of the first model', mse1)
mse2 = mean_squared_error(
    y_true=y2_test,
    y_pred=y_pred2,
)
print('MSE of the second model', mse2)

# The second model has a lower MSE, thus it does a better job predicting the MEDV than the univariate model.

# In general, the more features the model includes the lower the MSE would be. Yet be careful about including
# too many features. Some features could be random noise, thus hurt the interpretability of the model.
# In this case we used 2 features, from which we had RM having the highest positive correlation with MEDV,
# and LSTAT having the highest (lowest value) negative correlation with MEDV


