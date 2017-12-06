import score_funcs
import numpy as np
import Ant
import Analyze
import collections
from operator import itemgetter

'''This module contains the functionality for the Ant Colony Optimization Clustering algorithm.'''

def ACO(dataset, iterations, num_clusters, num_ants, beta, prob_cutoff, num_elite_ants, decay_rate, q, score_funcs):
    ''' The main function for the ACO algorithm. Takes in the dataset to be clustered, 
        maximum number of iterations, the number of ants to be included, and the score
        functions to be used. Creates individual ants, tracks the pheromone matrix,
        and updates the best clustering found so far. '''

    pheromone_matrix = _initialize_pheromones(dataset, num_clusters)
    ants = _initialize_ants(dataset, num_ants, num_clusters, beta, prob_cutoff, pheromone_matrix)
    best_score = iteration_best_score = float("inf")
    best_clustering = None
    results = []

    #_print_ant_info(ants)

    for iteration in range(iterations):

        print("-------------------------------------------------------------------")
        print ("Iteration: " + str(iteration + 1))

        for point_number in range(len(dataset)):

            for i, ant in enumerate(ants):
                        
                ant.update_beliefs() #Could potentially parallelize this step

        #After all data points have been classified for all ants, rank the ants by objective function
        rank_info = _rank_ants(ants)
        ants = [ranked_ant[0] for ranked_ant in rank_info.ants_and_scores]
        
        #Let the elite (best scoring) ants update the pheromone matrix, then update ants' matrices
        pheromone_matrix = _update_pheromones(pheromone_matrix, ants[0:num_elite_ants], decay_rate, q)
        _update_ants_pheromones(pheromone_matrix, ants)

        iteration_best_score = rank_info.best_score
        iteration_best_clustering = rank_info.best_clustering
        #results.append(iteration_best_score)

        if iteration_best_score < best_score:
            best_score = iteration_best_score
            best_clustering = iteration_best_clustering

        #Reset the ants' memory lists
        _reset_ants(ants)

        result = Analyze.analyze_clusters(best_clustering, score_funcs)
        # print("In CL result: %s" % str(result))
        results.append(result)

    return results


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
    ''' Rank the ants (best to worst) in terms of the cluster sse objective function. 
        Return the best score and the best clustering found this iteration. '''

    iteration_best_score = float("inf")
    iteration_best_clustering = None
    ranked_ants = []
    ant_rank = collections.namedtuple('ant_rank', ['best_score', 'best_clustering', 'ants_and_scores'])

    for ant in ants:
        clusters = ant.get_clustering()
        sse = score_funcs.cluster_sse(clusters)
        ranked_ants.append((ant, sse))

        #If best score see
        if sse < iteration_best_score:
            iteration_best_score = sse
            iteration_best_clustering = clusters

    ranked_ants = sorted(ranked_ants, key=itemgetter(1))

    return_info = ant_rank(iteration_best_score, iteration_best_clustering, ranked_ants)
    return return_info
    

def _update_pheromones(pheromones, ants, decay_rate, q):
    ''' Update the pheromone matrix based on the newly ranked ants
        using the ant quantity system pheromone update. '''
    
    updated_pheromones = []

    #Decay the existing pheromones
    for i in range(len(pheromones)):
        updated_pheromones.append(np.multiply(pheromones[i], decay_rate))

    #Add the new pheromones from elite ants
    for ant in ants:

        for i, data_point in enumerate(ant.dataset):

            for j, centroid in enumerate(ant.centroids):

                dist = np.linalg.norm(np.array(centroid) - np.array(data_point))
                pheromone_delta = q/dist
                updated_pheromones[j][i] += pheromone_delta

    return updated_pheromones

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
    #data = [[15, 26, 13], [.023, .222, .999], [13, 22, 16], [.015, .322, .897]]
    #ACO(data, iterations = 15, num_clusters = 2, num_ants = 2, beta = 0.5, prob_cutoff = 0.75, num_elite_ants = 1)

    file = open("iris.data", "r")
    data_lines = file.readlines()
    data = []
    for line in data_lines:
        
        data_line = line.split(",")[0:4]
        if len(data_line) > 2:
            data.append(data_line)

    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = float(data[i][j])


    results = ACO(data, iterations = 10, num_clusters = 3, num_ants = 8, beta = 0.5, 
        prob_cutoff = 0.75, num_elite_ants = 3, decay_rate = .75, q = 1, score_funcs = scores)
    for result in results:
        print(result)    