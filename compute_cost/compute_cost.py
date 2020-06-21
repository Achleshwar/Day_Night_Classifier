import numpy as np

def compute_cost(AL,Y):
	"""
	Implement cost function

	Input:
	"AL" : contains the prediction values
	"Y"  : true "label" vector (for example--> day : 0, night : 1)

	Returns: 
	"cost" : cross - entropy cost
	"""

	m = Y.shape[1]

	cost = (-1/m)*(np.sum(np.multiply(np.log(AL), Y)) + np.sum(np.multiply(np.log(1 - AL), 1-Y)))

	cost = np.squeeze(cost)

	return cost