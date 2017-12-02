import random
import numpy as np
import Analyze


def _weighted_choice(weights):
    rnd = random.random() * sum(weights)
    # rnd = np.random.uniform(low=0,high=(sum(weights)-1))
    for i, w in enumerate(weights):
        rnd -= w
        if rnd < 0:
            return i


def _calc_distance(vector1, vector2):
    """Return the distance of ||vector1 - vector2||
    """
    return np.linalg.norm(np.subtract(vector1, vector2))


def _calc_weight(centroids, data_point):
    min_dist = float("inf")
    for center in centroids:
        dist = _calc_distance(center, data_point)
        if dist < min_dist:
            min_dist = dist
    return min_dist * min_dist


def _initializeCentroids(k, data_set):
    """Initializes k centroids with point_dim dimensions. Uses the
    k-means++ method.
    """
    centroids = []
    # first one is just a random point in the dataset:
    centroids.append(random.choice(data_set))
    for i in range(1, k):
        # weights = map(_calc_weight, data_set)
        weights = [_calc_weight(centroids, point) for point in data_set]
        centroids.append(data_set[_weighted_choice(weights)])
    return centroids


def _hasConverged(old_centroids, new_centroids):
    for i in range(len(new_centroids)):
        if np.all(new_centroids[i] != old_centroids[i]):
            return False
    return True


def _group_points(dataset, centroids):
    clusterVectors = {}
    # for j in range(len(centroids)):
    #     clusterVectors.append([])
    for i in range(len(centroids)):
        clusterVectors[i] = []

    for vector in dataset:
        min_dist = float("inf")
        cluster = -1
        for i, center in enumerate(centroids):
            dist = _calc_distance(vector, center)
            if dist < min_dist:
                min_dist = dist
                cluster = i
        clusterVectors[cluster].append(vector)
    return clusterVectors


def _findMeanVectors(cluster_dict, data_set):
    """Find the mean vectors for each cluster, then return  that vector
    """
    mean_vectors = []
    for i, cluster in cluster_dict.items():
        if cluster == []:
            mean_vectors.append(np.random.choice(data_set))
        else:
            mean = np.mean(cluster, axis=0)
            mean_vectors.append(mean)
    return mean_vectors


def kMeans(data_set, score_funcs, k):
    assert(k <= len(data_set))
    results_list = []
    old_centroids = _initializeCentroids(k, data_set)
    clusters = _group_points(data_set, old_centroids)

    results_list.append(Analyze.analyze_clusters(clusters, score_funcs))

    new_centroids = _findMeanVectors(clusters, data_set)
    while not _hasConverged(old_centroids, new_centroids):
        old_centroids = new_centroids
        clusters = _group_points(data_set, new_centroids)
        # keep recording the data
        results_list.append(Analyze.analyze_clusters(clusters, score_funcs))
        new_centroids = _findMeanVectors(clusters, data_set)
    # last item is a repeat:
    return results_list[:-1]
