import numpy as np
import Layer
import collections
import score_funcs
import Analyze

'''This module contains the functionality for the competitive learning clustering algorithm.'''


def competitive_learning(data_set, eta, num_clusters, iterations, score_funcs):
    ''' The main competitive learning algorithm. Creates a two layer network,
        then trains the weights of the network by updating the weights of the
        node with the strongest output for each training example '''

    #Initialize variables
    num_inputs = len(data_set[0])  # Number of inputs is equal to the number of features
    weight_layer = Layer.Layer(num_inputs, num_clusters, eta)
    results = []

    for iteration in range(iterations):

        #Train the network, score the resulting clustering, append the score
        #to the list of scores, and move on to next iteration
        weight_layer = _train_network(data_set, weight_layer, num_clusters)
        clustering = _cluster(data_set, weight_layer)
        result = Analyze.analyze_clusters(clustering, score_funcs)
        results.append(result)

    return results


def _train_network(data_set, weight_layer, num_clusters):
    ''' Given the data set and the current network weights, run one iteration
        of weight updates for each example in the data set '''

    winners_seen = []

    #For each data point, select the best node, update the weights
    #of the "winning" node, and keep track of all clusters selected
    for data_point in data_set:

        winner_node = _select_cluster(data_point, weight_layer.weights)
        weight_layer.update_weights(winner_node, data_point)
        winners_seen.append(winner_node)

    # If a cluster hasn't been selected, randomize it's weights before the next round
    for i in range(num_clusters):
        if i not in winners_seen:
            weight_layer.randomize_weights(i, len(weight_layer.weights[i]))

    return weight_layer


def _select_cluster(data_point, weights):
    ''' Given a data point and the current weights, compute which cluster the
        data point belong to. '''

    best_score = -9999999.0
    selected_cluster = -1

    for i, weight_array in enumerate(weights):

        #Take the dot product of the input vector and the weight vector for
        #each output node in the network
        temp_score = np.dot(np.array(data_point), weight_array)

        #Track the best cluster seen so far
        if temp_score > best_score:
            best_score = temp_score
            selected_cluster = i

    return selected_cluster


def _cluster(data_set, weight_layer):
    ''' Return a clustering solution, based on the weights of the 
        competitive learning network. '''

    clustering = collections.defaultdict(list)

    for d in data_set: 
        cluster = _select_cluster(d, weight_layer.weights)
        clustering[cluster].append(d)

    return clustering
