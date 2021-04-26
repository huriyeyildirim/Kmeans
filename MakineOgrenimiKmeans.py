%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()  

# for plot styling

import numpy as np

from sklearn.datasets.samples_generator import make_blobs

X, y_true = make_blobs(n_samples=300, centers=4,
                       
cluster_std=0.60, random_state=0)

plt.scatter(X[:, 0], X[:, 1], s=50);

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4)

kmeans.fit(X)

y_kmeans = kmeans.predict(X)


plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],color="red")

plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],color="blue")

plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],color="orange")

plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],color="yellow")

centers = kmeans.cluster_centers_

plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

