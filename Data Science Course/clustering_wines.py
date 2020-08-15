# Clustering is a type of unsupervised learning that allows us to find groups of similar objects, objects that are
# more related to each other than to the objects in other groups. This is often used when we don’t have access to the
# ground truth, in other words, the labels are missing.

# Examples of business use cases include the grouping of documents, music, and movies based on their contents,
# or finding customer segments based on purchase behavior as a basis for recommendation engines.

# The goal of clustering is to separate the data into groups, or clusters, with more similar traits to each other
# than to the data in the other clusters.

# In general, there are four types:

# 1) Centroid based models - each cluster is represented by a single mean vector (e.g., k-means),
# 2) Connectivity based models - built based on distance connectivity (e.g., hierarchical clustering)
# 3) Distribution based models - built using statistical distributions (e.g., Gaussian mixtures)
# 4) Density based models - clusters are defined as dense areas (e.g., DBSCAN)

# In this module, we will explore the simple and widely-used clustering algorithm, k-means, to reveal subgroups of
# wines based on the chemical analysis reports.

# One of the most popular clustering algorithms is k-means. Assuming that there are n data points, the algorithm
# works as follows:

# Step 1: initialization - pick k random points as cluster centers, called centroids Step 2: cluster assignment -
# assign each data point to its nearest centroid based on its distance to each centroid, and that forms k clusters
# Step 3: centroid updating - for each new cluster, calculate its centroid by taking the average of all the points
# assigned to the cluster Step 4: repeat steps 2 and 3 until none of cluster assignments change, or it reaches the
# maximum number of iterations

# The K-Means algorithm has gained great popularity because it is easy to implement and scales well to large
# datasets. However, it is difficult to predict the number of clusters, it can get stuck in local optimums,
# and it can perform poorly when the clusters are of varying sizes and density. is

# How do we calculate the distance in k-means algorithm? One way is the euclidean distance, a straight line between
# two data points as shown below.

# For example, the euclidean distance between points x1 = (0, 1) and x2 = (2, 0) are given by:
# d = sqrt((x2-x1)^2 + (y2-y1)^2)
# d = sqrt(4 + 1) = 2.24

# One can extend it to higher dimensions. In the n-dimensional space, there are two points:
# p = (p1, p2, .. , pn)
# q = (q1, q2, .. , qn)

# Then the euclidean distance from p to q is given by the Pythagorean formula:
# d = sqrt((q1-p1)^2 + (q2-p2)^2 + .. + (qn-pn)^2)

# There are other distance metrics, such as Manhattan distance, cosine distance, etc. The choice of the distance
# metric depends on the data.

import numpy as np

p = np.array([5, 2])
q = np.array([6, 3])
d = np.sqrt((q - p) ** 2).sum()

# In this module, we analyze the result of a chemical analysis of wines grown in a particular region in Italy. And
# the goal is to try to group similar observations together and determine the number of possible clusters. This would
# help us make predictions and reduce dimensionality. As we will see there are 13 features for each wine,
# and if we could group all the wines into, say 3 groups, then it is reducing the 13-dimensional space to a
# 3-dimensional space. More specifically we can represent each of our original data points in terms of how far it is
# from each of these three cluster centers.

# The analysis reported the quantities of 13 constituents from 178 wines: alcohol, malic acid, ash, alcalinity of
# ash, magnesium, total phenols, flavanoids, nonflavanoid phenols, proanthocyanins, color intensity, hue, od280/od315
# of diluted wines, and proline.

import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# download the dataset
wine_dataset = load_wine()

# create the dataframe from the dataset
wine_df = pd.DataFrame(
    data=wine_dataset.data,
    columns=wine_dataset.feature_names
)

# (178, 13) -> 178 datapoints, each related to 13 features
print(wine_df.shape)

# For the ease of display, we show the basic statistics of the first 3 features using pandas iloc. # We use df.loc[
# 'key'] to search by left hand side feature (index) column, in our case the left hand side feature (index) column is
# index_col='name' Alternatively, if we do know the integer position(s), we can use .iloc to access the row(s).
print(wine_df.iloc[:, :3].describe())

# Another way to check for column names and the datatype of each column is to use .info().
print(wine_df.info())

# There are no missing values. It is worth noting that the attributes are not on the same scale. We will have to
# scale the data later.

# The summary statistics provide some of the information, while visualization offers a more direct view showing the
# distribution and the relationship between features.

# Here we introduce a plotting function to display histograms along the diagonal and the scatter plots for every pair
# of attributes off the diagonal, from pandas.plotting import 'scatter_matrix', for the ease of display, let’s show
# just two features (1st and 6th feature):
scatter_matrix(
    wine_df.iloc[:, [0, 5]]
)
plt.show()

# As we don’t know the ground truth, we look into the scatter plots to come up with a reasonable candidate for k,
# the number of clusters. There seem to be roughly three subgroups. Remember that there are no right or wrong answers
# for the number of subgroups. In the real world data, rarely do we find clear clusters; but we come up with our best
# educated guess. For example, in the scatter plot above, there seem to be three subgroups.

# No matter whether it is a supervised or unsupervised learning problem, exploratory data analysis (EDA) is essential
# and strongly recommended before one dives into modeling.

# After examining all the pairs of scatter plot, we pick two features to better illustrate the algorithm: alcohol and
# total_phenols, whose scatterplot also suggests three subclusters.
X = wine_df[
    ['alcohol', 'total_phenols']
].values
# Unlike any supervised learning models, in general, unsupervised machine learning models do not require to split
# data into training and testing sets since there is no ground truth to validate the model. However, centroid-based
# algorithms require one pre-processing step because k-means works better on data where each attribute is of similar
# scales. One way to achieve this is to standardize the data; mathematically: z = (x - mean) / std, where x is the
# raw data, mean and std are the average and standard deviation of x, and z is the scaled x such that it is centered
# at 0 and it has a unit standard deviation. StandardScaler under the sklearn.preprocessing makes it easy.

# Why and Where to Apply Feature Scaling? Real world dataset contains features that highly vary in magnitudes, units,
# and range. Normalisation should be performed when the scale of a feature is irrelevant or misleading and not should
# Normalise when the scale is meaningful. The algorithms which use Euclidean Distance measure are sensitive to
# Magnitudes. Here feature scaling helps to weigh all the features equally. Formally, If a feature in the dataset is
# big in scale compared to others then in algorithms where Euclidean distance is measured this big scaled feature
# becomes dominating and needs to be normalized. Examples of Algorithms where Feature Scaling matters 1. K-Means uses
# the Euclidean distance measure here feature scaling matters. 2. K-Nearest-Neighbours also require feature scaling.
# 3. Principal Component Analysis (PCA): Tries to get the feature with maximum variance, here too feature scaling is
# required. 4. Gradient Descent: Calculation speed increase as Theta calculation becomes faster after feature
# scaling. Note: Naive Bayes, Linear Discriminant Analysis, and Tree-Based models are not affected by feature
# scaling. In Short, any Algorithm which is Not Distance based is Not affected by Feature Scaling.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

plt.scatter(
    x=X[:, 0],
    y=X[:, 1],
)
plt.scatter(
    x=X_scaled[:, 0],
    y=X_scaled[:, 1],
)
plt.xlabel('alcohol')
plt.ylabel('total_phenols')
plt.legend(['X', 'X_scaled'])
plt.show()

# We show the original (red) and scaled (blue) data in the plot to visualize the effect of scaling. After scaling,
# the data is centered around (0, 0), and the ranges along x- and y-axis are roughly the same, from -2.5 to 2.5.

# It is a good practice to scale the features before training the model if the algorithms are distance-based

# Just like linear regression and k nearest neighbours, or any machine learning algorithms in scikit-learn,
# to do the modeling, we follow instantiate / fit / predict workflow. There are other arguments in KMeans,
# such as method to initialize the centroids, stopping criteria, etc., yet we focus on the number of clusters,
# n_clusters, and allow other parameters to take the default values. Here we specify 3 clusters:
kmeans_model = KMeans(
    n_clusters=3
)
kmeans_model.fit(X_scaled)
y_pred = kmeans_model.predict(X_scaled)

# There are 53 wines in cluster 0, 65 in cluster 1, and 60 in cluster 2
unique, counts = np.unique(y_pred, return_counts=True)
freq_dict = dict(zip(unique, counts))
print(freq_dict.items())

# To inspect the coordinates of the three centroids:
# we know that we have n_clusters=3, and each center of the centroid has coordinates [x, y]
print(kmeans_model.cluster_centers_)

# A better way to see the results is to visualize them:
# plot the scaled data
plt.scatter(
    x=X_scaled[:, 0],
    y=X_scaled[:, 1],
    c=y_pred,
)
# plot the centroids centers
plt.scatter(
    x=kmeans_model.cluster_centers_[:, 0],
    y=kmeans_model.cluster_centers_[:, 1],
    c=[0, 1, 2],
    marker='s',
    edgecolors='black',
    s=150,
)
plt.xlabel('alcohol')
plt.ylabel('total_phenols')
plt.title('KMeans, k=3')

# The stars are the centroids. K-means divides wines into three groups: low alcohol but high total phenols (upper
# right in green), high alcohol and high total phenols (upper left in yellow), and low total phenols (bottom in
# purple). For any new wine with the chemical report on alcohol and total phenols, we now can classify it based on
# its distance to each of the centroids. Suppose that there is new wine with alcohol at 13 and total phenols at 2.5,
# let’s predict which cluster the model will assign the new wine to.

# Create a 2d dataset
# we do not need to put the values attribute here, because it is not a DataFrame, but a numpy array already!
X_new = np.array([
    [13, 2.5]
])
X_new_scaled = scaler.fit_transform(X_new)
new_pred = kmeans_model.predict(X_new_scaled)
print(new_pred)
plt.scatter(
    x=X_new_scaled[0, 0],
    y=X_new_scaled[0, 1],
    color='red',
    label='new data point'
)
plt.legend()
plt.show()

# One major shortcoming of k-means is that the random initial guess for the centroids can result in bad clustering,
# and k-means++ algorithm addresses this obstacle by specifying a procedure to initialize the centroids before
# proceeding with the standard k-means algorithm. In scikit-learn, the initialization mechanism is set to k-means++,
# by default.

# Elbow Method
# Can we divide the wines into two subgroups? Yes!
# Can we divide the wines into four subgroups? Sure!
# As shown, k-means will be happy to divide the dataset into any integer number of clusters, ranging from 1, an extreme
# case where all data points belong to one big cluster, to n, another extreme case where each data point is its own
# cluster.
# So which one should we choose, 2, or 3, or 4 for the wines?

# Intuitively, k-means problem partitions n datapoints into k tight sets such that the data points are closer to each
# other than to the data points in the other clusters. And the tightness can be measured as the sum of squares of the
# distance from data point to its nearest centroid, or inertia.
# In scikit-learn, it is stored as inertia_, e.g. when k = 2, the distortion is 185:

print("Inertia (sum of squares of the distances between each point and center of the cluster's centroid):",
      kmeans_model.inertia_)

# calculate distortion for a range of number of clusters:
inertia_arr = []
for i in range(1, 11):
    km_model = KMeans(
        n_clusters=i
    )
    km_model.fit(X_scaled)
    inertia_arr.append(km_model.inertia_)

print(inertia_arr)

# For example, k=3 seems to be optimal, as we increase the number of clusters from 3 to 4, the decrease in inertia
# slows down significantly, compared to that from 2 to 3. This approach is called elbow method (can you see why?). It
# is a useful graphical tool to estimate the optimal k in k-means.

# plot the results on the elbow plot
plt.plot(
    np.arange(1, 11),
    inertia_arr,
    marker='o',
    color='black'
)
plt.vlines(
    x=3,
    ymin=np.min(inertia_arr),
    ymax=np.max(inertia_arr),
    color='red',
    linestyle='--',
    label='optimal number of clusers'
)
plt.xlabel('n_clusters')
plt.legend()
plt.ylabel('inertia')
plt.title('Inertia, n_features=2')
plt.show()

# One single inertia alone is not suitable to determine the optimal k because the larger k is, the lower the inertia
# will be.

# Previously to build kmeans models, we used two (out of thirteen) features: alcohol and total phenols. The choice is
# random and it is easy to visualize the results. However, can we use more features, for example all of them? Why
# not? Let’s try it.
X_all = wine_df.values
X_all_scaled = scaler.fit_transform(X_all)

inertia_all = []
for i in range(1, 11):
    km_model = KMeans(
        n_clusters=i
    )
    km_model.fit(X_all_scaled)
    inertia_all.append(km_model.inertia_)

# plot the results on the elbow plot
plt.plot(
    np.arange(1, 11),
    inertia_all,
    marker='o',
    color='black'
)
plt.vlines(
    x=3,
    ymin=np.min(inertia_all),
    ymax=np.max(inertia_all),
    color='red',
    linestyle='--',
    label='optimal number of clusers'
)
plt.xlabel('n_clusters')
plt.ylabel('inertia')
plt.legend()
plt.title('Inertia, n_features={}'.format(X_all_scaled.shape[1]))
plt.show()

# Similarly we spot that the inertia no longer decreases as rapidly after k = 3. We then finalize the model by
# setting n_clusters = 3 and obtain the predictions.
k_optimal = 3
final_kmeans = KMeans(
    n_clusters=k_optimal
)
final_kmeans.fit(X_all_scaled)
all_pred = final_kmeans.predict(X_all_scaled)
print(y_pred)
print(all_pred)
print(
    (y_pred != all_pred).sum()
)

# It is natural to ask, which model is better?

# Recall that clustering is an unsupervised learning method, which indicates that we don’t know the ground truth of
# the labels. Thus it is difficult, if not impossible, to determine that the model with 2 features is more accurate
# in grouping wines than the one with all 13 features, or vice versa. Which model, in other words which features,
# should you choose is often determined by external information. For example, the marketing department wants to know
# if a continent-specific strategy is needed to sell these wines. We now have access to consumers' demographic
# information and the three clusters identified from model A correspond better to customers in Europe, Asia,
# and North America respectively than model B; then model A is the winner. It is an oversimplified example,
# but you get the gist.
# In practice, the features are often chosen by the collaboration between data scientists and domain knowledge experts.

