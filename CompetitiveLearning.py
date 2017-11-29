import numpy as np
import Layer

'''This module contains the functionality for the competitive learning clustering algorithm.'''

def competitive_learning(data_set, eta, num_clusters, iterations):
	''' The main competitive learning algorithm. Creates a two layer network,
		then trains the weights of the netork by updating the weights of the 
		node with the strongest output for each training example '''

	num_inputs = len(data_set[0]) #Number of inputs is equal to the number of features
	weight_layer = Layer.Layer(num_inputs, num_clusters, eta)

	for iteration in range(iterations):

		weight_layer = _train_network(data_set, weight_layer)

	_cluster(data_set, weight_layer)

def _train_network(data_set, weight_layer):
	''' Given the data set and the current network weights, run one iteration
		of weight updates for each example in the data set '''

	for data_point in data_set:

		winner_node = _select_cluster(data_point, weight_layer.weights)
		weight_layer.update_weights(winner_node, data_point)

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

	for d in data_set: 
		cluster = _select_cluster(d, weight_layer.weights)
		print (d)
		print (cluster)


if __name__ == '__main__':
	
	data = [[5.6, 0.15, 4.9], [4.7, 0.12, 5.75], [0.22, 3.1, 0.007], [.35, 4.01, 0.23], [43.5, 6.7, 0.1], [51.2, 7.1, 0.25]]
	competitive_learning(data, 0.1, 4, 200)