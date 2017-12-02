import score_funcs
import numpy as np
import Ant

'''This module contains the functionality for the Ant Colony Optimization Clustering algorithm.'''

def ACO(dataset, iterations, num_clusters, num_ants, beta, score_funcs):
    ''' The main function for the ACO algorithm. Takes in the dataset to be clustered, 
        maximum number of iterations, the number of ants to be included, and the score
        functions to be used. Creates individual ants, tracks the pheramone matrix,
        and updates the best clustering found so far. '''

    pheramone_matrix = _initialize_pheramones(dataset, num_clusters)
    ants = _initialize_ants(dataset, num_ants, num_clusters)

def _initialize_pheramones(dataset, num_clusters):
    ''' Given the dataset and the number of clusters to be produced, initialize the 
        pheramone matrix (dimensions of size of dataset by number of clusters) to small
        values. '''

    pheramone_matrix = []
    num_instances = len(dataset)

    for i in range(num_clusters):
        cluster_pheramones = np.random.uniform(0, 0.2, size=num_instances)
        pheramone_matrix.append(cluster_pheramones)

    return pheramone_matrix

def _initialize_ants(dataset, num_ants, num_clusters, beta):
    ''' Initialize the population of ants. '''

    ants = []

    for i in range(num_ants):

        ant = Ant.ant(dataset, num_clusters, beta)
        ants.append(ant)

    return ants
