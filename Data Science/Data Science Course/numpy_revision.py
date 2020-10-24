import numpy as np

heights = [189, 170, 189, 163, 183, 171, 185, 168, 173, 183, 173, 173, 175, 178, 183, 193, 178, 173, 174, 183, 183,
           180, 168, 180, 170, 178, 182, 180, 183, 178, 182, 188, 175, 179, 183, 193, 182, 183, 177, 185, 188, 188,
           182, 185, 191]

print(
    len([i for i in heights if i > 180])
)

# array containing booleans if height > 188
are_tall = [i> 188 for i in heights]
# array containing all heights > 188

tall = [i for i in heights if i > 188]
print(sum(tall))

# No matter the format, the first step in data science is to transform it into arrays of numbers.
height_arr = np.array(heights)

# show number of heights > 188
# without the need to create a list using the conditional list comprehension
print(np.sum(height_arr > 188))

# Once an array is created in numpy, its size cannot be changed.

# Size tells us how big the array is, shape tells us the dimension. To get current shape of an array use attribute
# shape.
print(height_arr.size, height_arr.shape)

ages = [57, 61, 57, 57, 58, 57, 61, 54, 68, 51, 49, 64, 50, 48, 65, 52, 56, 46, 54, 49, 51, 47, 55, 55, 54, 42, 51,
        56, 55, 51, 54, 51, 60, 62, 43, 55, 56, 61, 52, 69, 64, 46, 54, 47, 70]

heights_and_ages = heights + ages

# Convert it to 1D numpy array
height_age_arr = np.array(heights_and_ages)

# We have 2 arrays containing 45 items
# Let's reshape it so 1st row contains heights and 2nd ages
height_age_arr = height_age_arr.reshape(2, height_age_arr.size//2)
print(height_age_arr.shape)

# Numpy can calculate the shape (dimension) for us if we indicate the unknown dimension as -1. For example,
# given a 2darray `arr` of shape (3,4), arr.reshape(-1) would output a 1darray of shape (12,), while arr.reshape((-1,
# 2)) would output a 2darray of shape(6, 2)

# Remember that 1D array containing 3 elements has shape (3,). After reshaping it to reshape(1, -1) (1 row,
# -1 remaining columns) we would obtain (1, 3)

height_age_arr = height_age_arr.reshape(-1)
print(height_age_arr.shape)
# Reshape to 2 rows and -1 columns, meaning that all datapoints are distributed into 2 rows and each row will contain
# equal number of datapoints (columns) = n // 2
height_age_arr = height_age_arr.reshape(2, -1)
print(height_age_arr.shape)

# Numpy supports several data types such as int (integer), float (numeric floating point), and bool (boolean values,
# True and False). The number after the data type, ex. int64, represents the bitsize of the data type.
print(height_age_arr.dtype)

height_age_arr_float = height_age_arr.astype(float)
print(height_age_arr_float.dtype)

# Get the 2nd row (1) and 1st column (0) in the 2D array
print(height_age_arr[1, 0])

# Numpy slicing syntax follows that of a python list: arr[start:stop:step]. When any of these are unspecified,
# they default to the values start=0, stop=size of dimension, step=1.

# Show 3rd and 4th (without 5th!) elements in all 2 rows
print(height_age_arr[:, 3:5])

# It is easy to update values in a subarray when you combine arrays with slicing.
height_age_arr[:, :10] = 0
print(height_age_arr[:, :10])

# Set height and age to specific value in a certain range
# Set height to 200 and age to 30 for 1st president, 0th column in the 2D array
height_age_arr[:, 0] = 200, 30
print(height_age_arr[:, :10])

# Updating a multidimensional array with a new record is straightforward in numpy as long as their shapes match.
height_age_arr[:, 1] = [190, 40]
print(height_age_arr[:, :10])

height_age_arr[:, 5:8]= [
    [190, 195, 200], [25, 30, 40]
    ]
print(height_age_arr[:, :10])

new_record = np.array([
    [175, 180, 185, 190],
    [35, 36, 37, 38]
    ])
height_age_arr[:, 41:] = new_record
print(height_age_arr[:, 40:])

# hstack and vstack visualisation
# hstack (horizontal, by column):
# [ [a, b], [a, b], [a, b]]
# vstack (vertical by row):
# [ [a, a, a], [b, b, b]]

height_arr = np.array(heights)
age_arr = np.array(ages)

# Reshape these 1D arrays, so they are 2D arrays containing 45 rows and single column
height_arr = height_arr.reshape(45, 1)
age_arr = age_arr.reshape(45, 1)

# Stack these arrays horizontally, so the final 2D array contains 45 rows and 2 columns
height_age_arr = np.hstack(
    (height_arr, age_arr)
)
print(height_age_arr[:5, :])

# Now stack them vertically
height_arr = height_arr.reshape(1, 45)
age_arr = age_arr.reshape(1, 45)

# Stack these arrays vertically, so the final 2D array contains 2 rows and 45 columns
height_age_arr = np.vstack(
    (height_arr, age_arr)
)
print(height_age_arr[:, :5])

# We can also use the concatenation of the arrays
# axis=0 is for stacking vertically y
# (each row is corresponding to feature)
# [[a a a a a a a a a]
#  [b b b b b b b b b]]

# axis=1 is for stacking horizontally x
# (each column is corresponding to feature)
# [[a b]
#  [a b]
#  [a b]]

height_arr = height_arr.reshape(45, 1)
age_arr = age_arr.reshape(45, 1)

height_age_arr = np.concatenate(
    (height_arr, age_arr),
    axis=1
)
print(height_age_arr[:5, :])

height_arr = height_arr.reshape(1, 45)
age_arr = age_arr.reshape(1, 45)
height_age_arr = np.concatenate(
    (height_arr, age_arr),
    axis=0
)
print(height_age_arr[:, :5])

# Change the values to meters
# Note that we obtain a 1D array containing only the 1st row (heights) of height_age_arr!
height_in_meters = height_age_arr[0, :] / 100
print(height_in_meters[:5])

# Operations on numpy arrays
# Other operations, such as .min(), .max(), .mean(), work in a similar way to .sum().
print('Sum of first 5 heights:',
    np.sum(height_age_arr[:, :5], axis=1)[0]
    )

# Comparisons
print('Presidents who started their cadency after the age of 45:', [i for i in height_age_arr[1, :] if i >= 45])
print((height_age_arr[1, :] >= 45).sum())

# Masking
height_age_arr = np.concatenate(
    (height_arr.reshape(45, 1), age_arr.reshape(45, 1)),
    axis=1
)

mask1 = (height_age_arr[:, 0] > 182) & (height_age_arr[:, 1] > 45)
# In case of using the Horizontal Stacking (by columns), we use the axis=1, and the mask should be set to [mask1, ]
tall_and_old = height_age_arr[mask1, ]
print(tall_and_old)

height_age_arr = np.concatenate(
    (height_arr.reshape(1, 45), age_arr.reshape(1, 45)),
    axis=0
)
mask2 = (height_age_arr[0, :] < 182) & (height_age_arr[1, :] < 50)
# In case of using the Vertical Stacking (by rows), we use the axis=0, and the mask should be set to [:, mask2]
small_and_young = height_age_arr[:, mask2]
print(small_and_young)
