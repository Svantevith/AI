# The pandas library is built on top of numpy, meaning a lot of features, methods, and functions are shared.
# As numpy ndarrays are homogeneous, pandas relaxes this requirement and allows for various dtypes in its data structures.

import pandas as pd
import numpy as np

# Series is a one-dimensional labeled array that can hold data of any type (integer, string, float, python objects, etc.), similar to a column in an excel spreadsheet. The axis labels are collectively called index.
# Think of Series as numpy 1darray with index or row names.
series_1 = pd.Series(
    [1, 2, 3],
    index=['a', 'b', 'c']
)

series_2 = pd.Series(
    np.array([1, 2, 3]),
    index=['a', 'b', 'c']
)

series_3 = pd.Series(
    {'a': 1,
     'b': 2,
     'c': 3}
)

arr_n = [1, 2, 3]
arr_l = ['a', 'b', 'c']
dict_ = dict(zip(arr_l, arr_n))

series_4 = pd.Series(dict_)

print(
    series_1.equals(series_2)
    and series_2.equals(series_3)
    and series_3.equals(series_4),
    '\n'
)

# In data science, data is usually more than one-dimensional, and of different data types; thus Series is not sufficient. DataFrames are 2darrays with both row and column labels.
# Think of DataFrame as a collection of the Series.

wine_dict = {
    'red_wine': [5, 8, 3],
    'white_wine': [2, 4, 0]
}

df = pd.DataFrame(
    wine_dict,
    index=['Bob', 'Mia', 'Joe']
)
# shows the data connected with feature (column label)
print(df['white_wine'], '\n')

# index_col will specify the left hand side column, in this case index_col='name' can be changed to height, age or party
# index_col is the index column, which contains indexes (keys) for each row
presidents_df = pd.read_csv('https://sololearn.com/uploads/files/president_heights_party.csv', index_col='name')
# print(presidents_df)

# by default n=5
print(presidents_df.head(n=3))
print(presidents_df.tail(n=3))

# use the attributes shape and size to obtain the dimentions tuple and number of elements
print('Number of rows:', presidents_df.shape[0])
print('Total number of elements', presidents_df.size, '\n')

# Use info() method to get an overview of the DataFrame. Its output includes index, column names, count of non-null values, dtypes, and memory usage.

# In addition to shape and size as seen in numpy, pandas features allow additional functionality to examine the data.
presidents_df.info()

# We use df['key'] to search by feature (column)
print('\n', presidents_df['party'])
# We use df.loc['key'] to search by left hand side feature (index) column, in our case the left hand side feature (index) column is index_col='name'
print("\nResults for df.loc['Abraham Lincoln']", presidents_df.loc['Abraham Lincoln'])

print('\nAll presidents from Abraham Lincoln till Ulysses S. Grant:\n',
      presidents_df.loc['Abraham Lincoln': 'Ulysses S. Grant'])

# We can also get data for specific index columns (left hand side column)
print('\nData for specific index columns:\n', presidents_df.loc[
    ['Abraham Lincoln', 'Zachary Taylor', 'Barack Obama']
])

# Alternatively, if we do know the integer position(s), we can use .iloc to access the row(s).
print('\nPresidents using df.iloc[15:18]:\n', presidents_df.iloc[15:18])

# When accessing a single column, one bracket results in a Series (single dimension) and double brackets results in a DataFrame (multi dimensional).
# In case of a single key:
print('\nHeights of first 5 presidents:\n', presidents_df['height'].head())

# In case of multiple keys use the list of keys:
print('\nHeights and ages of last 5 presidents:\n', presidents_df[
    ['height', 'age']
].tail()
      )

# We can get search using both index column (rows keys, left hand side column) and by column names (upper column names)

# Remember that using loc with keys, the range between keys is inclusive, while iloc has exlusive right limit
# Notice that using loc, the range 'key1':'key2' we don't use square brackets [], while using multiple keys (specific positions) we use the list of keys and a single key or range is without sqaure brackets []
# Select all rows from index column using loc[:, ...]
print('\nResults for 3 desired presidents, showing the columns from order to height:',
      presidents_df.loc[
      ['Abraham Lincoln', 'Zachary Taylor', 'Barack Obama'], 'order': 'height'])

# Measures of Location - Minimum, Maximum, Mean

# Measures of Spread - Range, Variance, Standard Deviation

# min, max, mean, std, var, range, median, quantile methods etc. may be used on whole detasets or specifically chosen
# parts of DataFrame and Series If no rows and columns are selected, these method will return a DataFrame/Series
# containing computed statistical values for each column (considering data from all rows)

# describe() prints out almost all of the summary statistics mentioned previously except for the variance. In
# addition, it counts all non-null values of each column. use include='all' to view all Null and non-numerical values
print('\nPresidents DataFrame description:\n', presidents_df.describe(include='all'))
print('\nSummary Statistics for age:\n', presidents_df['age'].describe())

# Remember that we search for data using the column keys in that way df[column key] or df[[column keys]]
# and using the index column (keys for rows) using the df.loc[row key] or df.loc[[row keys]]
# There is also possibility to use loc for first purpose using : to access all rows
print('\nQuantiles for the Age column:\n', presidents_df['age'].quantile([0.25, 0.5, 0.75, 1.]))

print('\nMax height of first 3 presidents:\n', presidents_df[:3]['height'].max())

# The fourth column 'party' was omitted in the output of .describe() because it is a categorical variable (numbers
# such as height, age, order are the numerical variables that are used for numerical calculations). A categorical
# variable is one that takes on a single value from a limited set of categories. It doesn’t make sense to calculate
# the mean of democratic, republican, federalist, and other parties.

# categorical variables are basically string value columns in the dataset. We can use label encoding and one hot
# encoding to convert categorical variables to numbers.

# We can check the unique values and corresponding frequency (in descending order) by using .value_counts():
print('\nUnique values of party and their frequency:\n', presidents_df['party'].value_counts())
print('\nSummary statistics for party:\n', presidents_df['party'].describe())

# To find the value based on a condition, we can use the groupby operation. Think of groupby doing three steps:
# split, apply, and combine. The split step breaks the DataFrame into multiple DataFrames based on the value of the
# specified key; the apply step is to perform the operation inside each smaller DataFrame; the last step combines the
# pieces back into the larger DataFrame.

# The .groupby("party") returns a DataFrameGroupBy object, not a set of DataFrames. To produce a result,
# apply an aggregate (.mean()) to this DataFrameGroupBy object:
print('\nShow the minimum values for the DataFrame grouped by party:\n', presidents_df.groupby('party').min())

# We can also perform multiple operations on the groupby object using .agg() method. It takes a string, a function,
# or a list thereof. For example, we would like to obtain the min, median, and max values of heights grouped by party:
print('\nMin, Median, Max values of heights grouped by party:\n',
      presidents_df.groupby('party')['height'].agg(['min', np.median, max]))

# Often time we are interested in different summary statistics for multiple columns. For instance, we would like to
# check the median and mean of heights, but minimum and maximum for ages, grouped by party. In this case, we can pass
# a dict with key indicate the column name, and value indicate the functions:
print('\nMedian and Mean values for height, Min and Max for ages, grouped by party:\n',
      presidents_df.groupby('party').agg({'height': [np.median, np.mean], 'age': [np.min, np.max]}))

# Recall that mean is more sensitive to any change in dataset. For example, median may not change after adding a
# single value, while mean will instantly change.