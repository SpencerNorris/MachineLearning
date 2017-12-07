#!/usr/bin/env python3

'''
Author: Spencer Norris
File: neural_net.py
Description: implementation of a 2-input,
1-output neural network for classification
with two hidden units in a single hidden 
layer. Network is trained using gradient 
descent on the squared in-sample error.
'''

import matplotlib.pyplot as plt
from copy import copy
import numpy as np
import random
import math
import sys


def __plot_decision_boundary(D, test_predictions):
	#Separate test points into +1s, -1s
	ones = np.array([ 
		D[i, 1:] 
		for i in range(len(D))
		if test_predictions[i] == 1
	])
	not_ones = np.array([ 
		D[i, 1:] 
		for i in range(len(D))
		if not test_predictions[i] == 1
	])

	print("Ones: ", len(ones))
	print("Not Ones: ", len(not_ones))

	#Plot points using original 2-D coordinates
	fig, ax = plt.subplots()
	ax.scatter(ones[:,0], ones[:,1], color='b', alpha=.2)
	ax.scatter(not_ones[:,0], not_ones[:,1], color='r', alpha=.2)
	ax.set_xlabel("Anti-Symmetry")
	ax.set_ylabel("Avg. Intensity")
	plt.show()
	plt.close()


def __get_symmetry_score(x):
	'''
	Takes the sums of the squared errors between the pixel values
	flipped along the X axis and Y axis and mapped onto the other side.
	Since it increases with the difference when mirrored across the X axis,
	a better term might be the 'anti-symmetry'
	'''
	#Reshape data into matrix
	dim = int(math.sqrt(len(x)))
	x = np.reshape(x, (dim,dim))

	#Find midpoint(s)
	top = bottom = int(len(x) / 2)
	if top % 2 == 0:
		bottom -= 1

	#Iterate over matrix, get pairwise squared difference
	score = 0.0
	while top < int(len(x)):
		for i in range(len(x.T) - 1):
			score += (x[top][i] - x[int(len(x) / 2) + bottom][i])**2
		bottom -= 1
		top += 1
	return score


def __get_avg_intensity(x):
	'''
	The average pixel value for the image.
	'''
	return sum(x) / len(x)


def sigmoid(x):
	x = float(x)
	if x == 0:
		return .5
	return 1.0 / (1.0 + math.exp(-x))


def linear(x):
	return x


class NeuralNet():
	def __init__(self,
				 size_in=2,
				 size_hidden=2, 
				 learning_rate=.001,
				 epochs=1,
				 activation_fn=sigmoid,
				 output_fn = np.tanh,
				 variable_learning_rate=False):

		self.epochs = epochs
		self.learning_rate = learning_rate
		self.activation_fn = activation_fn
		self.output_fn = output_fn
		self.size_in = size_in
		self.size_hidden = size_hidden
		self.variable_learning_rate = variable_learning_rate

		#Input-to-hidden weights (Include extra weight for the bias terms)
		if '--one' in sys.argv:
			self.input_hidden_weights = np.array([[.25 for j in range(size_hidden)] for i in range(size_in + 1)])
			self.hidden_output_weights = np.array([[.25 for j in range(1)] for i in range(size_hidden + 1)])
		else:
			self.input_hidden_weights = np.array([[random.uniform(-.25,.25) for j in range(size_hidden)]
											for i in range(size_in + 1)])
			self.hidden_output_weights = np.array([[random.uniform(-.25,.25) for j in range(1)] 
											for i in range(size_hidden + 1)])


	def __feedforward(self, X, training=False):

		#Reshape X, append bias term
		Xin = [[x] for x in X]
		Xin = np.concatenate(([[1]], Xin), axis=0)
		node_signals = []
		node_outputs = []

		#Propogate values through network
		input_hidden_signals = np.matmul(self.input_hidden_weights.T, Xin)
		hidden_outputs = np.array([[sigmoid(signal)] for signal in input_hidden_signals])
		hidden_outputs = np.concatenate(([[1.0]], hidden_outputs), axis=0)
		hidden_output_signal = np.matmul(
								self.hidden_output_weights.T, 
								hidden_outputs
							)[0][0]
		output_value = self.output_fn(hidden_output_signal)

		node_signals.append(Xin)
		node_signals.append(input_hidden_signals)
		node_signals.append(hidden_output_signal)

		node_outputs.append(Xin)
		node_outputs.append(hidden_outputs)
		node_outputs.append(output_value)

		if training:
			return output_value, node_signals, node_outputs
		else:
			return output_value


	def train(self, X, Y):
		'''
		Wrapper function for training the network
		using the feedforward function with backpropogation.
		'''
		#Perturb all weights by .0001 if called for
		if '--perturb' in sys.argv:
			self.input_hidden_weights = self.input_hidden_weights + np.full(
												(len(self.input_hidden_weights), 
											 	len(self.input_hidden_weights.T)),
												.0001
											)
			self.hidden_output_weights = self.hidden_output_weights + np.full(
												(len(self.hidden_output_weights), 
											 	len(self.hidden_output_weights.T)),
												.0001
											)

		#Backpropogation with batch gradient descent
		E_in = []
		D = np.concatenate(([[y] for y in Y], X), axis=1)
		for epoch in range(self.epochs):
			np.random.shuffle(D)
			results = []
			for example in D:
				res, node_signals, node_outputs = self.__feedforward(example[1:], True)
				results.append(res)

			#Calculate E_total
			orig_E_total = .25 * (1.0 / float(len(results))) * sum([(results[i] - D[i,0])**2 for i in range(len(results))])
			if orig_E_total == 0.0:
				return

			gradient = []
			gradient.append(np.array([[0.0 for j in range(self.size_hidden)] for i in range(self.size_in + 1)]))
			gradient.append(np.array([[0.0 for j in range(1)] for i in range(self.size_hidden + 1)]))
			deltas = copy(gradient)

			for example in D:
				#Feedforward
				x = example[1:]
				res, node_signals, node_outputs = self.__feedforward(x, True)

				__output_deltas = np.array([0.0 for i in range(1)])
				__hidden_deltas = np.array([0.0 for i in range(self.size_hidden)])

				#Deltas for output
				if self.output_fn is np.tanh:
					output_derivative = lambda x: 1 - node_outputs[2]**2
				else:
					output_derivative = lambda x: 1
				__output_deltas = 2 * (res - example[0]) * output_derivative(node_signals[2])

				#Deltas for hidden layer
				sig_derivative = lambda x: sigmoid(x) * (1 - sigmoid(x))
				sig_deriv_transform = np.array([ #Applies the derivative of the sigmoid function element-wise
										[sig_derivative(node_signals[1][i])] 
										for i in range(len(node_signals[1]))
									])
				__hidden_deltas = sig_deriv_transform * (self.hidden_output_weights[1:] * __output_deltas)
				
				#Update the gradient with deltas
				gradient[0] = gradient[0] + (np.outer(node_outputs[0].T[0], __hidden_deltas.T[0]) / float(len(X)))
				gradient[1] = gradient[1] + (np.outer(node_outputs[1].T[0], __output_deltas) / float(len(X)))

			#Update weights with gradient
			orig_input_hidden_weights = copy(self.input_hidden_weights)
			orig_hidden_output_weights = copy(self.hidden_output_weights)
			self.input_hidden_weights = self.input_hidden_weights - (self.learning_rate * gradient[0])
			self.hidden_output_weights = self.hidden_output_weights - (self.learning_rate * gradient[1])

			#If we're using variable learning rates, check how to adjust
			if self.variable_learning_rate:
				#Calculate E_total with updated weights
				alpha = 1.1
				beta = .9
				test_results = []
				for example in D:
					res, node_signals, node_outputs = self.__feedforward(example[1:], True)
					test_results.append(res)
				test_E_total = .25 * (1.0 / float(len(test_results))) * sum(
									[(test_results[i] - D[i,0])**2 for i in range(len(results))
								])

				if test_E_total < orig_E_total:
					self.learning_rate = self.learning_rate * alpha
					E_in.append(test_E_total)
				else:
					self.learning_rate = self.learning_rate * beta
					self.input_hidden_weights = orig_input_hidden_weights
					self.hidden_output_weights = orig_hidden_output_weights
					E_in.append(orig_E_total)
			else:
				pass

			if len(X) == 1:
				print("Gradient: ")
				print(gradient[0], gradient[1])
				print("Weights: ")
				print(self.input_hidden_weights)
				print(self.hidden_output_weights)

		#Plot the in-sample versus training iteration
		if self.variable_learning_rate:
			plt.plot(range(self.epochs), E_in, color='b')
			plt.show()
			plt.close()

	def predict(self, X):
		return np.sign(
			np.array([self.__feedforward(x, False) for x in X])
		)


def main():
	if '--one' in sys.argv:
		D = np.array([[1.0,1.0,1.0]])
		X = np.array(D[:,:2])
		Y = np.array(D[:,2])
		epochs = 1
	else:
		#Read in data sets, extract x and y
		training = np.genfromtxt('ZipDigits.train.txt')
		test = np.genfromtxt('ZipDigits.test.txt')
		D = np.concatenate((training,test), axis=0)
		X = D[:, 1:]
		Y = D[:,0]

		#Modify Y so that digits that aren't 1 are -1
		Y = [1 if y == 1 else -1 for y in Y]

		#Apply feature transforms to X, normalize symmetry (intensity is already (-1,+1)).
		#Normalization performed by shifting -1, scaling with diff between max and min.
		X = np.array([[__get_symmetry_score(x), __get_avg_intensity(x)] for x in X])
		symm_min = min(X[:,0])
		symm_max = max(X[:,0])
		X = np.array([[2*((x[0] - symm_min) / (symm_max - symm_min)) - 1, x[1]] for x in X])

		#Re-attach Y to X, assign to D, Shuffle dataset and partition
		D = np.concatenate(([[y] for y in Y], X), axis=1)
		np.random.shuffle(D)

	size_in = len(X.T)

	if '--hidden' in sys.argv:
		size_hidden = int(sys.argv[sys.argv.index('--hidden') + 1])
	else:
		size_hidden = 2

	if '--linear' in sys.argv:
		output_fn = linear
	else:
		output_fn = np.tanh

	if '--eta' in sys.argv:
		learning_rate = float(sys.argv[sys.argv.index('--eta') + 1])
	else:
		eta = .0001

	if '--epochs' in sys.argv:
		epochs = int(sys.argv[sys.argv.index('--epochs') + 1])
	else:
		epochs = 1

	if '--variable' in sys.argv:
		variable_learning_rate = True
	else:
		variable_learning_rate = False

	#Train neural net
	net = NeuralNet(
				size_in=size_in,
				size_hidden=size_hidden, 
				learning_rate=eta, 
				activation_fn=sigmoid,
				output_fn=output_fn,
				epochs=epochs,
				variable_learning_rate=variable_learning_rate
			)
	net.train(D[:300,1:], D[:300,0])

	#Predict test set, plot
	predictions = net.predict(D[300:,1:])
	__plot_decision_boundary(D[300:], predictions)



if __name__ == '__main__':
	sys.exit(main())