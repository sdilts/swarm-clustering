import numpy as np
import Layer
import collections

'''This module contains the functionality for the competitive learning clustering algorithm.'''

def competitive_learning(data_set, eta, num_clusters, iterations, score_funcs):
    ''' The main competitive learning algorithm. Creates a two layer network,
        then trains the weights of the netork by updating the weights of the 
        node with the strongest output for each training example '''

    num_inputs = len(data_set[0]) #Number of inputs is equal to the number of features
    weight_layer = Layer.Layer(num_inputs, num_clusters, eta)
    results = []

    for iteration in range(iterations):

        weight_layer = _train_network(data_set, weight_layer, num_clusters)
        clustering = _cluster(data_set, weight_layer)
        result = Analyze.analyze_clusters(clustering, score_funcs)
        results.append(result)

    return results

def _train_network(data_set, weight_layer, num_clusters):
    ''' Given the data set and the current network weights, run one iteration
        of weight updates for each example in the data set '''

    winners_seen = []

    for data_point in data_set:

        winner_node = _select_cluster(data_point, weight_layer.weights)
        weight_layer.update_weights(winner_node, data_point)
        winners_seen.append(winner_node)

    #If a cluster hasn't been selected, randomize it's weights before the next round
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

        temp_score = np.dot(np.array(data_point), weight_array)
        if temp_score > best_score:
            best_score = temp_score
            selected_cluster = i

    return selected_cluster

def _cluster (data_set, weight_layer):

    clustering = collections.defaultdict(list)

    for d in data_set: 
        cluster = _select_cluster(d, weight_layer.weights)
        clustering[cluster].append(d)

    return clustering


if __name__ == '__main__':
    
    data = [[5.6, 0.15, 4.9], [4.7, 0.12, 5.75], [0.22, 3.1, 0.007], [.35, 4.01, 0.23], [43.5, 6.7, 0.1], [51.2, 7.1, 0.25]]
    c = competitive_learning(data, 0.1, 3, 200)

    #print (c)

    for item, val in c.items():
        print (str(item) + ": " + str(val))
