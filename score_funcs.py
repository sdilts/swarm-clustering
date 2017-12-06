import numpy as np

'''This module provides the summed squared error metric to evaluate clusters'''


def cluster_sse(clusters):
    ''' The cluster SSE evaluation method. This method calculates the summed
        squared error, measured by the distance between the centroid of each
        cluster and each of the points within that cluster. '''

    total_sse = 0

    # Find centroids of the provided clusters
    centroids = _calculate_centroids(clusters)

    # Sum up the individual cluster sse values
    for i, cluster in clusters.items():
        total_sse += _single_cluster_sse(cluster, centroids[i])

    return total_sse
cluster_sse.name = "Cluster SSE"


def _single_cluster_sse(cluster, centroid):
    # Calculate the SSE for a single cluster.

    sse = 0

    for point in cluster:

        # Calculate distance(cluster, point)^2
        squared_error = (np.linalg.norm(centroid - point))**2
        sse += squared_error

    return sse


def silhouette_coefficient(clusters):
    ''' The silhouette coefficient evaluation method. This method combines
        cohesion and separation by calculating the average distance from the
        centroid of each cluster to the data points within that cluster as
        well as the minimum separation of each cluster. These are combined to
        a single measure. '''

    s_c = 0
    centroids = _calculate_centroids(clusters)

    for i, cluster in clusters.items():

        avg_dist = _average_distance(cluster, centroids[i])
        separation = _minimum_separation(centroids, i)

        print ("AVG dist: " + str(avg_dist))
        print ("Separation: " + str(separation))

        if avg_dist < separation:
            coefficient = (separation - avg_dist) / separation
        else:
            coefficient = (separation - avg_dist) / separation

        s_c += coefficient


    return s_c
silhouette_coefficient.name = "Silhouette Coefficient"


def _average_distance(cluster, centroid):
    # Given a cluster and that cluster's centroid, calculate the average
    # distance between the centroid and all points in the cluster

    total_distance = 0

    for point in cluster:

        dist = np.linalg.norm(centroid - point)
        total_distance += dist

    return total_distance/len(cluster)  # Average distance


def _minimum_separation(centroids, i):
    # For centroid i, find the distance to the nearest centroid

    min_dist = float("inf")

    for j, centroid in centroids.items():

        if j != i:  # Don't want to consider the current centroid, distance would be zero
            dist = np.linalg.norm(centroids[i] - centroid)

            if dist < min_dist:
                min_dist = dist

    return min_dist


def _calculate_centroids(clusters):
    # Given a clustering, compute the centroids for each cluster.

    centroids = dict()

    for i, cluster in clusters.items():

        centroid = np.mean(cluster, axis=0)
        centroids[i] = centroid
    return centroids
