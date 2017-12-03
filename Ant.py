import random
import numpy as np

''' The ant class represents an individual ant in the ACO algorithm. Each ant holds
    its own beliefs about the weights of the clusterings, which indicate membership 
    of each data point to a cluster. An ant also holds an array of centroids, which 
    are based on that ant's weight matrix. Finally, given the ant's beliefs, the ant
    lays down pheromones, which influence the clustering behavior of the other ants
    in the population. '''

class ant:

    def __init__(self, dataset, num_clusters, beta):

        self.num_data_points = len(dataset)
        self.num_clusters = num_clusters
        self.beta = beta
        self.weights = self._initialize_weights(len(dataset))
        self.centroids = self.calculate_centroids(dataset)
        self.memory_list = []
        self.reset_memory()

    def reset_memory(self):
        ''' Reset the memory list at the beginning of each iteration. '''

        memory = []

        for i in range(self.num_data_points):

            memory.append(i)

        random.shuffle(memory)
        self.memory_list = memory

    def select_next_object(self):
        ''' Pop the next data point off the memory list. '''

        return self.memory_list.pop()

    def _initialize_weights(self, data_length):
        ''' Randomly initialize the ant's weight matrix. '''

        weights = []

        for i in range(data_length):

            weight_list = [0] * self.num_clusters
            cluster = random.randint(0, self.num_clusters - 1)
            weight_list[cluster] = 1
            weights.append(weight_list)

        return weights

    def calculate_centroids(self, dataset):
        ''' Based on the weight matrix, and given the dataset, calculate
            the centroids for the ant. '''

        clusters = []

        for i in range(self.num_clusters):

            cluster = []

            for j in range(len(dataset)):

                if self.weights[j][i] == 1:
                    cluster.append(dataset[j])

            clusters.append(cluster)

        centroids = {}

        for i, clust in enumerate(clusters):

            if clust != []:
                centroid = np.mean(clust, axis=0)
                centroids[i] = centroid

        return centroids

    def _heuristic_value(self, data_point, centroid):
        ''' Given a data point and a centroid, calculate the inverse Euclidean distance. '''

        dist = np.linalg.norm(centroid - point)
        return (1.0 / dist)


if __name__ == '__main__':

    data = [[1, 2, 3], [.023, .222, .999], [3, 2, 6], [.015, .322, .897]]
    test_ant = ant(data, 2, 0.5)
    obj = test_ant.select_next_object()