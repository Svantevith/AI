# matplotlib.pyplot is a collection of functions that make plotting in python work like MATLAB. Each function makes
# some change to a figure, e.g., creates a figure, creates a plotting area in a figure, plots lines, annotates the
# plots with labels, etc. as we will see in the following lessons.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# For all matplotlib plots, first create a figure and an axes object; to show the plot, call “plt.show()”. The figure
# contains all the objects, including axes, graphics, texts, and labels. The axes is a bounding box with ticks and
# labels. Think of axes as an individual plot.
fig = plt.figure()
ax = plt.axes()

# ####LINE PLOT#### np.linspace(0, 10, 1000) creates a range of 1000 linearly (evenly) distributed points in the
# range from 0 to 1000, it returns a 1D numpy array

# A line plot displays data along a number line, a useful tool to track changes over short and long periods of time.
x = np.linspace(0, 10, 1000)
y = np.sin(x)
ax.plot(x, y)
plt.show()

# Alternatively, we can use the pylab interface and let the figure and axes be created for us in the background.
# Instead of
# fig = plt.figure()
# ax = plt.axes()
# Just write:
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.xlim(0, 1000)
plt.ylim(-1, 1)
plt.title('Sin(x) for x in range [0;1000]')
plt.show()

# we can plot two functions on one figure
# instead of '--', '-.', etc. we can use 'solid', 'dashed', etc.
plt.plot(x, np.sin(x), color='k', linestyle='--')
plt.plot(x, np.cos(x), color='r', linestyle='-.')
plt.show()

# color = 'tab:red' for the Tableau palette
# color = 'red' works similarly
# opacity 'alpha= 0.7'

# Also we can give different style and color together using options

# gives red color dotted line with * marker
plt.plot(x, np.sin(x), 'r*--', label='sin(x)', linewidth=1.25)
# gives blue color normal line with triangle marker
plt.plot(x, np.cos(x), 'b^-', label='cos(x)', alpha=0.7)
# plt.legend() shows the legend if the labels are given to each plot
plt.legend()
plt.show()
# There are a lot of visual modification to the plots which can be easily found in documentation

# ####SCATTER PLOT####
presidents_df = pd.read_csv('https://sololearn.com/uploads/files/president_heights_party.csv', index_col='name')

# Scatter plot for each president assuming number of datapoints a number of indexes (index_col='name')
# For each index in the DataFrame (index_col='name'), there is plotted each president's age and height
plt.scatter(
    presidents_df['height'],
    presidents_df['age'],
    marker='<',
    color='orange'
    )
# scatter plot can be also obtained by plt.plot() where the style is for.example '<' left sided triangle arrow
plt.xlabel('Height')
plt.ylabel('Age')
plt.title('Correlation between age and height of U.S Presidents')
plt.show()

# Scatter plot is a useful tool to display the relationship between two features;
# whereas a line plot puts more emphasis on the change between points as its slope depicts the rate of change.

# A great thing about pandas is that it integrates well with matplotlib, so we can plot directly from DataFrames and
# Series. We specify the kind of plot as 'scatter', 'height' along x-axis, and 'age' along y-axis, and then give it a
# title. As we specified the x-axis and y-axis with column names from the DataFrame, the labels were also annotated
# on the axes.
presidents_df.plot(
    kind='scatter',
    x='height',
    y='age',
    title='Correlation between age and height of U.S Presidents'
)
plt.show()

# ####HISTOGRAM#### A histogram is a diagram that consists of rectangles with width equal to the interval and area
# proportional to the frequency of a variable.

# In a histogram, the range of x values is divided equally into a number (default 10) bins. Each bin is a rectangular
# bar with the left side aligned with a marker on the x-axis and the right one aligned with the next marker. Its
# height (the y value) represents the frequency of occurence of the values within the range (left marker to the right
# one) of that particular bin. For example, we have 3 bins and our total range is 0-99. So we have 4 markers (0, 33,
# 66, 99) and the bins have the following ranges: 1st bin: 0-33        2nd bin: 33-66       3rd bin: 66-99 Now,
# if in our dataset there are 25 datapoints with value between 0 and 33, 5 with that between 33 and 66, and 1 between
# 66 and 99, Then the first bin's top will have y = 25, 2nd will have y = 5 and 3rd 1.

# A histogram shows the underlying frequency distribution of a set of continuous data, allowing for inspection of the
# data for its shape, outliers, skewness, etc.

# pandas.DataFrame.plot by default set ylabel of histogram to 'frequency'
presidents_df['height'].plot(
    kind='hist',
    title='Histogram for heights column',
    bins=5
)
plt.xlabel('Height')
plt.show()

# is exactly the same as
plt.hist(
    presidents_df['height'],
    bins=5
)
plt.title('Histogram for heights column')
plt.xlabel('Height')
plt.ylabel('Frequency')

# ####BOX PLOT#### The corresponding boxplot is shown below. The blue box indicates the interquartile range (IQR,
# between the first quartile and the third), in other words, 50% data fall in this range. The red bar shows the
# median, and the lower and upper black whiskers are minimum and maximum.

# The box-and-whisker plot doesn’t show frequency, and it doesn’t display each individual statistic, but it clearly
# shows where the middle of the data lies and whether the data is skewed.

# As the red bar, the median, cuts the box into unequal parts, it means that the height data is skewed.
plt.style.use('classic')
presidents_df.boxplot(
    column='height'
)
plt.show()

# ####BAR PLOT#### Bar plots show the distribution of data over several groups. For example, a bar plot can depict
# the distribution of presidents by party.

# Bar plots are commonly confused with a histogram. Histogram presents numerical data whereas bar plot shows
# categorical data. The histogram is drawn in such a way that there is no gap between the bars, unlike in bar plots.
party_cnt = presidents_df['party'].value_counts()
plt.style.use('ggplot')
party_cnt.plot(
    kind='bar'
)
plt.show()

