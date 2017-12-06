import Analyze
import multiprocessing as mp
import numpy as np
from collections import deque
from collections import defaultdict
import time
import random
# import matplotlib.pyplot as plt

'''This module contains the functionality for the DBSCAN clustering algorithm.'''


# Test if point satisfies core point requirements
def core_point(index, all_points, radius, minpts):
    # Calculate the differences between the dimensions of the current point and all other dataset points
    current_point = all_points[index]
    differences = np.array(current_point) - np.array(all_points)

    # Calculate the euclidean distance between points
    # If point count exceeds or equals minpts, this point is core, return the point with its neighborhood
    point_count = 0
    neighborhood = []
    for i in range(len(all_points)):
        dist = np.linalg.norm(differences[i])

        if dist < radius:
            point_count += 1
            neighborhood.append(all_points[i])

    # Return the core point, along with the points in its neighborhood
    if point_count >= minpts:
        return current_point, neighborhood

    return None


def _cluster(data_points, radius, minpts):
    # Calculate which pts are core using multiple processes for speed
    pool = mp.Pool(processes=4)
    results = [pool.apply_async(core_point, args=(x, data_points, radius, minpts)) for x in range(len(data_points))]
    core_pts = [p.get() for p in results]

    # Add core pts to a dictionary for fast membership check, with point as key, and neighbors as values
    core_set = {}
    for pt in core_pts:
        if pt is not None:
            core_set[pt[0]] = pt[1]

    # Each pt receives a label assigning it to a cluster or noise
    cluster_label = {}
    for pt in data_points:
        cluster_label[pt] = "unknown"

    # Assign pts to cluster, starting with a core pt
    label_num = 0
    for core_pt, neighbor in core_set.items():
        # If core pt is unknown assign it to a new cluster
        if cluster_label[core_pt] == "unknown":
            label_num += 1
            cluster_label[core_pt] = label_num
            queue = deque()
            queue.append(core_pt)

        # When a new core point is encountered in the neighborhood of another core point it is added to a queue
        # In this way all "density reachable" points from the initial core pt are added to the current cluster
        while len(queue) > 0:
            current_point = queue.popleft()

            # Add core point neighbors to the current cluster
            for n in core_set[current_point]:
                # If one of the neighbor points is also a core point, add it to the queue
                if n in core_set and cluster_label[n] == "unknown":
                    queue.append(n)
                if cluster_label[n] == "unknown":
                    cluster_label[n] = label_num

    return cluster_label


# Method to determine a reasonable radius to use
def parameter_selection(k, data_pts):
    k_dist = []
    for pt in data_pts:
        distances = []
        differences = np.array(pt) - np.array(data_pts)
        for diff in differences:
            dist = np.linalg.norm(diff)
            distances.append(dist)
        distances.sort()
        k_dist.append(distances[k-1])

    k_dist.sort()
    x = range(0, len(k_dist))
    plt.scatter(x, k_dist)
    plt.show()


# Load data from a local file
def load_data(file_name):
    cluster = []
    with open(file_name, 'r') as file:
        for line in file:
            a = line.split(",")
            pt = (float(a[0]), float(a[1]))
            cluster.append(pt)
    print("Number of instances : %s" % len(cluster))
    return cluster


# Plot the clusters
def dbscan(data_pts, radius, minpts, score_funcs=None):
    labels = _cluster(data_pts, radius, minpts)

    clusters = defaultdict(list)
    for key, label in labels.items():
        clusters[label].append(key)

    result = [Analyze.analyze_clusters(clusters, score_funcs)]

    # cluster_color = {}
    # c = ['r', 'b', 'g', 'y', 'c', 'b']
    # x = []
    # y = []
    # color = []
    # for key, cluster in labels.items():
    #     if cluster is not "unknown":
    #         x.append(key[0])
    #         y.append(key[1])
    #
    #         if cluster not in cluster_color:
    #             cur = c.pop(0)
    #             cluster_color[cluster] = cur
    #             c.append(cur)
    #
    #         color.append(cluster_color[cluster])
    #
    # plt.scatter(x, y, c=color)
    # plt.show()

    return result
