import random
import numpy as np
from collections import defaultdict
import sys
import Analyze
import score_funcs

'''The Particle class contains the main functionality for the PSO algorithm, including fitness evaluation
   velocity updates, and position updates'''


class Particle:
    def __init__(self, k, w, c1, c2, data):
        self.c2 = c2                        # Social acceleration coefficient
        self.c1 = c1                        # Cognitive acceleration coefficient
        self.w = w                          # Inertia coefficient
        self.k = k                          # Number of clusters to find
        self.position = []                  # A particle position is a list of centroid vectors
        self.velocity = []                  # Initialize velocity vector to 0
        self.fitness_val = sys.maxsize      # Track current fitness of the particle

        # Initialize clusters by assigning random data pts from the data
        for i in range(k):
            rand_index = random.randint(0, len(data) - 1)
            centroid = np.array(data[rand_index])
            self.position.append(np.array(centroid))
            self.velocity.append(np.zeros(len(centroid)))

        # Initialize personal best to initial position
        self.personal_best = self.position

    # Method to update particle position, velocity, and personal best
    def update_particle(self, global_best, data):
        personal_best_fitness = self.fitness(data, self.personal_best)

        # If the fitness of the updated position is better than the personal best fitness, update the personal best
        cur_fitness = self.fitness(data, self.position)
        self.fitness_val = cur_fitness

        if cur_fitness < personal_best_fitness:
            self.personal_best = self.position

        self.update_velocity(global_best)
        self.update_position()

    # Method to update particle velocity
    def update_velocity(self, global_best):
        phi_1 = random.uniform(0, 1)
        phi_2 = random.uniform(0, 1)

        # print("Phi 1: %s" % phi_1)
        # print("Phi 2: %s\n" % phi_2)

        # Calculate the difference  between personal best and current position
        local_comp = [self.personal_best[i] - self.position[i] for i in range(self.k)]

        # Calculate difference between global best and current position
        global_comp = [global_best[i] - self.position[i] for i in range(self.k)]

        # print("Local comp: %s" % str(local_comp))
        # print("Global comp: %s" % str(global_comp))

        for i in range(len(self.velocity)):
            self.velocity[i] = self.w * self.velocity[i] + self.c1 * phi_1 * local_comp[i] + self.c2 * phi_2 * global_comp[i]

    # Method to update particle position
    def update_position(self):
        self.position = [self.position[i] + self.velocity[i] for i in range(len(self.position))]

    # Method to assess the fitness of a particle
    def fitness(self, data, position):
        # Assign points to a cluster
        cluster_assignments = defaultdict(list)
        for pt in data:
            pt = np.array(pt)
            # Calculate distance to each centroid in the cluster
            dist_to_cluster = [Particle.distance(pt, k) for k in self.position]

            # Assign the point to the closest centroid
            cluster_label = dist_to_cluster.index(min(dist_to_cluster))
            cluster_assignments[cluster_label].append(pt)

        # Evaluate fitness
        total = 0
        for key in cluster_assignments.keys():
            cluster_sum = 0
            for data_pt in cluster_assignments[key]:
                # Add the distance between the data pt and current cluster centroid
                cluster_sum += Particle.distance(data_pt, position[key])/len(cluster_assignments[key])

            total += cluster_sum

        return total/self.k

    @staticmethod
    def distance(vec1, vec2):
        diff = vec1 - vec2
        dist = np.linalg.norm(diff)
        return dist

