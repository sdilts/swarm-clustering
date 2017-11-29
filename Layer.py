import numpy as np
from sklearn.preprocessing import Normalizer

''' The layer class holds the weights in between the two layers of the 
	Competative Learning network. There is also a method to update the
	weights associated with a selected output node. '''

class Layer:

	def __init__(self, num_inputs, num_clusters):

		self.weights = []

		for i in range(num_clusters):
			
			new_weights = np.random.uniform(-0.2, 0.2, size=num_inputs) # Randomize the weights

			#Normalize the weights
			scaler = Normalizer().fit(new_weights.reshape(1, -1))
			normalized_weights = scaler.transform(new_weights.reshape(1, -1))
			self.weights.append(normalized_weights)

	def update_weights(self, eta, cluster_number, data_point):
		''' Given eta, the number of the output node to udpate, and the 
			data point being considered, update the weights for the given
			node. Then normalize the updated weights. '''

		#Normalize the data
		data_array = np.array(data_point)
		data_scaler = Normalizer().fit(data_array.reshape(1, -1))
		normalized_data = data_scaler.transform(data_array.reshape(1, -1))

		#Update the weights
		sample = np.array(normalized_data)
		weight_change = eta * (sample - self.weights[cluster_number])
		self.weights[cluster_number] += weight_change

		#Normalize the weights
		weight_scaler = Normalizer().fit(self.weights[cluster_number].reshape(1, -1))
		normalized_weights = weight_scaler.transform(self.weights[cluster_number].reshape(1, -1))
		self.weights[cluster_number] = normalized_weights


if __name__ == '__main__':

	lay1 = Layer(3, 4)
	for weight in lay1.weights:
		print (weight)
	lay1.update_weights(0.5, 2, [0.12, 0.3, .05])

	print ()
	for weight in lay1.weights:
		print (weight)