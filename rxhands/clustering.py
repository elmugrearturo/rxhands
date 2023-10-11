from sklearn.cluster import DBSCAN, KMeans

from rxhands.geometry import centroid

import numpy as np


def dbscan_clustering(points):
    points = np.array(points)
    clustering = DBSCAN(eps=3, min_samples=2)
    clustering.fit(points)
    centroids = {}
    for i in range(clustering.labels_.shape[0]) :
        label = clustering.labels_[i]
        point = points[i]
        centroids[label] = centroids.get(label, [])
        centroids[label].append(point)
    for label in centroids:
        centroids[label] = centroid(centroids[label])
    return centroids
    
def kmeans_clustering(points, no_clusters=2):
    points = np.array(points)
    clustering = KMeans(no_clusters, n_init="auto")
    clustering.fit(points)
    centroids = {}
    for i in range(clustering.labels_.shape[0]) :
        label = clustering.labels_[i]
        point = points[i]
        centroids[label] = centroids.get(label, [])
        centroids[label].append(point)
    for label in centroids:
        centroids[label] = centroid(centroids[label])
    return centroids
    
