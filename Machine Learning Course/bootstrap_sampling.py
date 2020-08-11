# https://www.analyticsvidhya.com/blog/2020/02/what-is-bootstrap-sampling-in-statistics-and-machine-learning/

# In statistics, Bootstrap Sampling is a method that involves drawing of sample data repeatedly with replacement from
# a data source to estimate a population parameter

# Bootstrap sampling is used in a machine learning ensemble algorithm called bootstrap aggregating (also called
# bagging). It helps in avoiding overfitting and improves the stability of machine learning algorithms.

# In bagging, a certain number of equally sized subsets of a dataset are extracted with replacement. Then,
# a machine learning algorithm is applied to each of these subsets and the outputs are ensembled.

# Here are a few key benefits of bootstrapping:

# The estimated parameter by bootstrap sampling is comparable to the actual population parameter

# Since we only need a few samples for bootstrapping, the computation requirement is very less

# In Random Forest, the bootstrap sample size of even 20% gives a pretty good performance


import numpy as np
import random

# Gaussian distribution (population) of 10,000 elements with the population mean being 500

# normal distribution (Gaussian distribution)
# loc - Mean (“centre”, "peak") of the distribution.
# scale - standard deviation (spread or "width") of the distribution
# size - output shape
population = np.random.normal(
    loc=500.0,
    scale=1.0,
    size=10000
)
print(type(population))
print(np.mean(population))

# Now, we will draw 40 samples of size 5 from the distribution (population) and compute the mean for every sample

sample_mean_vals = []

# population is ndarray (1D numpy array), we need to convert it tolist() or to any set or sequence in order to pass
# it to random.sample() method
for i in range(40):
    single_sample = random.sample(population.tolist(), 5)
    avg = np.mean(single_sample)
    sample_mean_vals.append(avg)

print(np.mean(sample_mean_vals))

