import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (13, 8)
np.set_printoptions(suppress=True)
pd.set_option("display.precision", 5)


def plotDistribGraph(pdf):
    fig, a = plt.subplots(ncols=1, figsize=(16, 5))
    a.set_title("Distributions")
    for col in pdf.columns:
        sns.kdeplot(pdf[col], ax=a)
    plt.show()


def plotGraph(pdf, pscaled_df):
    fig, (a, b) = plt.subplots(ncols=2, figsize=(16, 5))
    a.set_title("Before scaling")
    for col in pdf.columns:
        sns.kdeplot(pdf[col], ax=a)
    b.set_title("After scaling")
    for col in pdf.columns:
        sns.kdeplot(pscaled_df[col], ax=b)
    plt.show()


def plotGraphAll(pdf, pscaled1, pscaled2, pscaled3):
    fig, (a, b, c, d) = plt.subplots(ncols=4, figsize=(16, 5))
    a.set_title("Before scaling")
    for col in pdf.columns:
        sns.kdeplot(pdf[col], ax=a)
    b.set_title("RobustScaler")
    for col in pscaled1.columns:
        sns.kdeplot(pscaled1[col], ax=b)
    c.set_title("MinMaxScaler")
    for col in pscaled2.columns:
        sns.kdeplot(pscaled2[col], ax=c)
    d.set_title("StandardScaler")
    for col in pscaled3.columns:
        sns.kdeplot(pscaled3[col], ax=d)
    plt.show()


np.random.seed(1)
NBROWS = 5000
df = pd.DataFrame({
    'A': np.random.normal(0, 2, NBROWS),
    'B': np.random.normal(5, 3, NBROWS),
    'C': np.random.normal(-3, 3, NBROWS),
    'D': np.random.chisquare(7, NBROWS),
    'E': np.random.beta(10, 2, NBROWS) * 40,
    'F': np.random.normal(6, 4, NBROWS)
})

plotDistribGraph(df)

scaler = StandardScaler()
keepCols = ['A', 'B', 'C']
scaled_df = scaler.fit_transform(df[keepCols])
scaled_df = pd.DataFrame(scaled_df, columns=keepCols)
plotGraph(df[keepCols], scaled_df)

scaler = MinMaxScaler()
keepCols = ['A', 'B', 'C']
scaled_df = scaler.fit_transform(df[keepCols])
scaled_df = pd.DataFrame(scaled_df, columns=keepCols)
plotGraph(df[keepCols], scaled_df)

scaler = MaxAbsScaler()
keepCols = ['A', 'B', 'C']
scaled_df = scaler.fit_transform(df[keepCols])
scaled_df = pd.DataFrame(scaled_df, columns=keepCols)
plotGraph(df[keepCols], scaled_df)

scaler = RobustScaler()
keepCols = ['A', 'B', 'E','F']
scaled_df = scaler.fit_transform(df[keepCols])
scaled_df = pd.DataFrame(scaled_df, columns=keepCols)
plotGraph(df[keepCols], scaled_df)

# Let’s just summarize the Feature Scaling techniques we just encountered:
# 1) Scaling features to a range, often between zero and one, can be achieved using MinMaxScaler or MaxAbsScaler.
# 2) MaxAbsScaler was specifically designed for scaling sparse data, RobustScaler cannot be fitted to sparse inputs,
# but you can use the transform method on sparse inputs.
# 3) If your data contains many outliers, scaling using the mean and variance of the data is likely to not work
# very well. In this case, you need to use RobustScaler instead.