import random
import numpy as np

''' The ant class represents an individual ant in the ACO algorithm. Each ant holds
    its own beliefs about the weights of the clusterings, which indicate membership 
    of each data point to a cluster. An ant also holds an array of centroids, which 
    are based on that ant's weight matrix. Finally, given the ant's beliefs, the ant
    lays down pheromones, which influence the clustering behavior of the other ants
    in the population. '''

class ant:

    def __init__(self, dataset, num_clusters, beta, prob_cutoff, pheromones):

        self.num_data_points = len(dataset)
        self.num_clusters = num_clusters
        self.beta = beta
        self.prob_cutoff = prob_cutoff
        self.pheromones = pheromones
        self.dataset = dataset
        self.weights = self._initialize_weights(len(dataset))
        self.centroids = self._calculate_centroids()
        self.memory_list = []
        self.reset_memory()

    def reset_memory(self):
        ''' Reset the memory list at the beginning of each iteration. '''

        memory = []

        #Add the data points back onto the memory list
        for i in range(self.num_data_points):

            memory.append(i)

        random.shuffle(memory) #Randomize the order in which the ant will cluster the data points
        self.memory_list = memory

    def update_beliefs(self):
        ''' Select a random data point, cluster it, update weights and centroids. '''

        data_index = self._select_next_object() #Grab the index of the next data point to evaluate
        new_cluster = self._select_cluster(data_index) #Assign the data point to a cluster

        #Update the weight matrix and recalculate centroids given the newly clustered data point
        new_weight_vector = [0] * self.num_clusters
        new_weight_vector[new_cluster] = 1
        self.weights[data_index] = new_weight_vector
        self.centroids = self._calculate_centroids()

    def get_clustering(self):
        ''' Return the clustering of the data, given the current weights '''

        clusters = {}

        for i in range(self.num_clusters):

            clust = []
            for j in range(len(self.dataset)):

                if self.weights[j][i] == 1: #Data point j is in cluster i
                    clust.append(self.dataset[j])

            clusters[i] = clust

        return clusters


    def _select_next_object(self):
        ''' Pop the next data point off the memory list. '''

        return self.memory_list.pop()

    def _initialize_weights(self, data_length):
        ''' Randomly initialize the ant's weight matrix. '''

        weights = []
        clusters_chosen = []

        for i in range(data_length):

            #Assign weight of 1 to a random cluster for each data point
            weight_list = [0] * self.num_clusters
            cluster = random.randint(0, self.num_clusters - 1)

            #Keep track of which clusters have been selected
            if cluster not in clusters_chosen:
                clusters_chosen.append(cluster)

            weight_list[cluster] = 1
            weights.append(weight_list)

        #Cannot have any empty clusters, if empty cluster exists, reinitialize
        for c in range(self.num_clusters):
            if c not in clusters_chosen:
                weights = self._initialize_weights(data_length)

        return weights

    def _calculate_centroids(self):
        ''' Based on the weight matrix, and given the dataset, calculate
            the centroids for the ant. '''

        clusters = []

        for i in range(self.num_clusters):
        #Based on the weights, divide the datapoints into their clusters

            cluster = []

            for j in range(len(self.dataset)):

                if self.weights[j][i] == 1:
                    cluster.append(self.dataset[j])

            clusters.append(cluster)

        centroids = {}

        for i, clust in enumerate(clusters):
        #Calculate the mean of each cluster, dealing with the case of an empty cluster

            if clust != []:
                centroid = np.mean(clust, axis=0)
                centroids[i] = centroid
            else:
                #No data points to calculate cluster, pick a randomly scaled mean of all points
                scaling_vector = np.random.uniform(0, 1, size=len(self.dataset[0]))
                temp_centroid = np.mean(self.dataset, axis=0) * scaling_vector
                centroids[i] = temp_centroid

        return centroids

    def _heuristic_value(self, data_point, centroid):
        ''' Given a data point and a centroid, calculate the inverse Euclidean distance
            to the power of beta. '''

        dist = np.linalg.norm(centroid - data_point) #Calculate distance between data point and centroid
        
        if dist == 0:
            #Cannot divide by 0 (if cluster has only one data point), return small value
            return 0.0000001
        else:
            return (1.0 / dist)**self.beta

    def _select_cluster(self, data_index):
        ''' Given a data point, return the cluster to which the data point belongs. '''

        prob = random.random()

        if prob < self.prob_cutoff:
            #Choose the best cluster
            return self._best_cluster(data_index)

        else:
            #Probabilistically select the cluster
            return self._probabilistic_cluster(data_index)

    def _best_cluster(self, data_index):
        ''' Select the cluster with the highest score '''
        best_cluster = -1
        best_score = -9999999

        for i in range(self.num_clusters):

            #Calculate score for the current cluster
            pheromone = self.pheromones[i][data_index]
            heuristic = self._heuristic_value(self.dataset[data_index], self.centroids[i])
            temp_score = pheromone * heuristic

            #Keep track of the best cluster seen so far
            if temp_score > best_score:
                best_score = temp_score
                best_cluster = i 

        return best_cluster

    def _probabilistic_cluster(self, data_index):
        ''' Probilistically determine to which cluster the data point belongs '''

        scores = []
        total_score = 0
        cummulative_prob = 0
        probabilities = []

        for i in range(self.num_clusters):

            #Calculate score for the current cluster
            pheromone = self.pheromones[i][data_index]
            heuristic = self._heuristic_value(self.dataset[data_index], self.centroids[i])
            temp_score = pheromone * heuristic

            scores.append(temp_score)
            total_score += temp_score

        #Calculate the cummulative probability for each cluster
        for score in scores:

            cummulative_prob += (score/total_score)
            probabilities.append(cummulative_prob)

        rand_prob = random.random()

        #Return the first cluster for which random_prob <= cummulative prob
        for i, p in enumerate(probabilities):
            if rand_prob <= p:
                return i
