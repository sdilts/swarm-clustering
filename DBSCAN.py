import multiprocessing as mp
import numpy as np
import copy
import time
import random
import matplotlib.pyplot as plt

# Test if point satisfies core point requirements
def core_point(index, all_points, radius, minpts):
    # Calculate the differences between the dimensions of the current point and all other dataset points
    other_points = copy.deepcopy(all_points)
    current_point = other_points.pop(index)

    differences = np.array(current_point) - np.array(other_points)

    # Calculate the euclidean distance between points
    # If point count exceeds or equals minpts, this point is core, return True
    point_count = 0
    neighborhood = []
    for i in range(len(other_points)):
        dist = np.linalg.norm(differences[i])

        if dist < radius:
            point_count += 1
            neighborhood.append(other_points[i])

    if point_count >= minpts:
        return current_point, neighborhood
    else:
        return None


def dbscan(data_points, radius, minpts):
    t0 = time.time()
    pool = mp.Pool(processes=4)
    results = [pool.apply_async(core_point, args=(x, data_points, radius, minpts)) for x in range(len(data_points))]
    core_pts = [p.get() for p in results]
    t1=time.time()
    print("Pool Time %s" % (t1-t0))
    #print(core_pts[0][1])
    #print()

    cluster_label = {}
    for pt in data_points:
        cluster_label[pt] = "unknown"
    #print(cluster_label)

    # Assign pts to cluster
    label_num = 0
    for pt in core_pts:
        if pt is not None:
            #print(cluster_label)
            #print()
            if cluster_label[pt[0]] == "unknown":
                label_num += 1
                cluster_label[pt[0]] = label_num

            for neighbor in pt[1]:
                if cluster_label[neighbor] == "unknown":
                    #print("Neighbor")
                    # print(neighbor)
                    cluster_label[neighbor] = label_num

    # t0 = time.time()
    # core = []
    # for i in range(len(data_points)):
    #     is_core = core_point(i, data_points, radius, minpts)
    #     core.append(is_core)
    #
    # t1 = time.time()
    # print("Sequential Time %s" % (t1-t0))
    # print(core)

    return cluster_label


# test_points = [(2,3), (4,6), (4,5), (5,2), (2,7)]
# mylist = [(random.randint(0, 9), random.randint(0, 9)) for k in range(500)]

cluster1 = [(random.uniform(0,1), random.uniform(0,1)) for k in range(50)]
cluster2 = [(random.uniform(5,9), random.uniform(5,7)) for k in range(50)]
cluster = cluster1 + cluster2

labels = dbscan(cluster, 3, 5)
print(labels)

x = []
y = []
color = []
for key, cluster in labels.items():
    x.append(key[0])
    y.append(key[1])
    color.append(cluster)

plt.scatter(x,y,c=color)
plt.show()