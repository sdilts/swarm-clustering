import multiprocessing as mp
import numpy as np
from collections import deque
import copy
import time
import random
import matplotlib.pyplot as plt


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

    if point_count >= minpts:
        return current_point, neighborhood
    else:
        return None


def dbscan(data_points, radius, minpts):
    # Calculate which pts are core using multiple processes for speed
    t0 = time.time()
    pool = mp.Pool(processes=4)
    results = [pool.apply_async(core_point, args=(x, data_points, radius, minpts)) for x in range(len(data_points))]
    core_pts = [p.get() for p in results]

    # Add core pts to a set for fast membership check
    core_set = set()
    for pt in core_pts:
        if pt is not None:
            core_set.add(pt[0])

    # print("Test Set")
    # print(core_set)
    t1=time.time()
    print("Pool Time %s" % (t1-t0))
    #print(core_pts[0][1])
    #print()

    # t0 = time.time()
    # core_pts = []
    # for i in range(len(data_points)):
    #     is_core = core_point(i, data_points, radius, minpts)
    #     core_pts.append(is_core)
    #
    # t1 = time.time()
    # print("Sequential Time %s" % (t1-t0))
    # print(core)

    cluster_label = {}
    for pt in data_points:
        cluster_label[pt] = "unknown"

    # Assign pts to cluster
    # cluster_colors = {}
    # colors = ['r', 'b', 'g', 'c', 'y', 'k']
    label_num = 0
    for pt in core_pts:
        if pt is not None:
            if cluster_label[pt[0]] == "unknown":
                label_num += 1
                # c = colors.pop(0)
                # cluster_colors[label_num] = c
                # colors.append(c)
                cluster_label[pt[0]] = label_num

            queue = deque(pt)
            while len(queue) > 0:
                current_point = queue.popleft()
                for neighbor in current_point[1]:
                    if neighbor in core_set:
                        queue.append()
                    if cluster_label[neighbor] == "unknown":
                        cluster_label[neighbor] = label_num

    return cluster_label # , cluster_colors


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


def load_data(file_name):
    cluster = []
    with open(file_name, 'r') as file:
        for line in file:
            a = line.split(",")
            pt = (float(a[0]), float(a[1]))
            cluster.append(pt)
    print("Number of instances : %s" % len(cluster))
    return cluster


def cluster(data_pts, radius, minpts):
    result = dbscan(data_pts, radius, minpts)
    labels = result[0]
    colors = result[1]
    print(colors)
    x = []
    y = []
    color = []
    for key, cluster in labels.items():
        if cluster is not "unknown":
            x.append(key[0])
            y.append(key[1])
            color.append(colors[cluster])

    plt.scatter(x,y,c=color)
    plt.show()