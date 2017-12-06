import Particle
import Analyze
import sys
import os
import numpy as np
from collections import defaultdict
# import matplotlib.pyplot as plt

'''This module contains the functionality to run the particle swarm optimization algorithm'''


def pso(num_particles, k, w, c1, c2, max_iter, data, score_funcs=None):
    swarm = []                       # List to hold particles of the swarm
    global_best = []                 # Global best position
    global_fitness = sys.maxsize     # Fitness value of the global best position
    g_fit = []                       # Fitness history of the global best position

    # Create particle swarm and initialize global best position
    for i in range(num_particles):
        cur_particle = Particle.Particle(k, w, c1, c2, data)
        swarm.append(cur_particle)

        # As new particles are created update the global best position
        particle_fitness = cur_particle.fitness(data, cur_particle.position)
        if particle_fitness < global_fitness:
            global_best = cur_particle.position
            global_fitness = particle_fitness

        g_fit.append(global_fitness)

    # print("init global best: %s, fitness: %s" % (str(global_best), str(global_fitness)))

    # Run particle swarm
    results_list = []
    for i in range(max_iter):
        # print("iteration: %s" % i)

        # print("init global best: %s" % (str(global_best)))

        # Update global fitness if a particle local fitness is better
        for particle in swarm:
            if particle.fitness_val < global_fitness:
                global_best = particle.position
                global_fitness = particle.fitness_val
        g_fit.append(global_fitness)

        clusters = assign_cluster(data, global_best)
        results_list.append(Analyze.analyze_clusters(clusters, score_funcs))

        # Update the positions and velocities of the particle
        for particle in swarm:
            particle.update_particle(global_best, data)

    # return global_best, g_fit
    # plt.plot(g_fit)
    # plt.show()
    return g_fit


# Method to assign data points to a cluster
def assign_cluster(data, global_best):
    clusters = defaultdict(list)
    for pt in data:
        pt = np.array(pt)
        dist_to_cluster = [distance(pt, k) for k in global_best]

        # Assign the point to the closest centroid
        cluster_label = dist_to_cluster.index(min(dist_to_cluster))
        clusters[cluster_label].append(pt)

    return clusters


def distance(vec1, vec2):
    diff = vec1 - vec2
    dist = np.linalg.norm(diff)
    return dist
