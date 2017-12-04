import score_funcs
import numpy as np
import Ant

'''This module contains the functionality for the Ant Colony Optimization Clustering algorithm.'''

def ACO(dataset, iterations, num_clusters, num_ants, beta, score_funcs, prob_cutoff):
    ''' The main function for the ACO algorithm. Takes in the dataset to be clustered, 
        maximum number of iterations, the number of ants to be included, and the score
        functions to be used. Creates individual ants, tracks the pheromone matrix,
        and updates the best clustering found so far. '''

    pheromone_matrix = _initialize_pheromones(dataset, num_clusters)
    ants = _initialize_ants(dataset, num_ants, num_clusters, beta, prob_cutoff, pheromone_matrix)

    print("Pheromones:")
    print(pheromone_matrix)
    print ("")

    _print_ant_info(ants)

    for iteration in range(iterations):

        print("-------------------------------------------------------------------")
        print ("Iteration: " + str(iteration + 1))

        for point_number in range(len(dataset)):

            for i, ant in enumerate(ants):
                        
                ant.update_beliefs() #Could potentially parallelize this step

        #After all data points have been classified for all ants update pheromones
        #and reset memory lists
        ants = _rank_ants(ants)
        pheromone_matrix = _update_pheromones(pheromone_matrix, ants)
        _update_ants_pheromones(pheromone_matrix, ants)
        _reset_ants(ants)

        _print_ant_info(ants)

    print ("------------------------------------------")
    print ("")

    for a in ants:
        clust = a.get_clustering()
        print (clust)


def _initialize_pheromones(dataset, num_clusters):
    ''' Given the dataset and the number of clusters to be produced, initialize the 
        pheromone matrix (dimensions of size of dataset by number of clusters) to small
        values. '''

    pheromone_matrix = []
    num_instances = len(dataset)

    for i in range(num_clusters):
        cluster_pheromones = np.random.uniform(0, 0.2, size=num_instances)
        pheromone_matrix.append(cluster_pheromones)

    return pheromone_matrix

def _initialize_ants(dataset, num_ants, num_clusters, beta, prob_cutoff, pheromones):
    ''' Initialize the population of ants. '''

    ants = []

    for i in range(num_ants):

        ant = Ant.ant(dataset, num_clusters, beta, prob_cutoff, pheromones)
        ants.append(ant)

    return ants

def _reset_ants(ants):
    ''' Reset the memory list for the ants. '''

    for ant in ants:
        ant.reset_memory()

def _rank_ants(ants):

    return ants

def _update_pheromones(pheromones, ants):
    ''' Update the pheromone matrix based on the newly ranked ants. '''
    
    return pheromones

def _update_ants_pheromones(pheromones, ants):
    ''' Update each individual ant's pheromone matrix. '''

    for ant in ants:
        ant.pheromones = pheromones

def _print_ant_info(ants):
    ''' For debugging. '''

    for i, ant in enumerate(ants):
        print ("Ant " + str(i))
        print ("weights: " + str(ant.weights))
        print ("centroids: " + str(ant.centroids))
        print ("")


if __name__ == '__main__':

    scores = [score_funcs.cluster_sse]
    data = [[1, 2, 3], [.023, .222, .999], [3, 2, 6], [.015, .322, .897]]
    ACO(data, iterations = 3, num_clusters = 2, num_ants = 2, beta = 0.5, score_funcs = scores, prob_cutoff = 0.75)
