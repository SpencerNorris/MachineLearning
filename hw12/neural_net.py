#!/usr/bin/env python3

'''
Author: Spencer Norris
File: neural_net.py
Description: implementation of a 1-output neural network
with linear output transform for regression, tanh output
transform for classification. Uses sum of squared errors as
error function. Parameters for run are specified
using command line flags as follows:

--decay : use weight decay during backpropogation.

--variable : use variable learning rates during backpropogation,
			with alpha = 1.1 and beta = .9

--early-stopping : will use early stopping with a validation
				`set of 50 data points randomly selected from
				the training set during each training epoch;
				the remaining 250 examples are used for
				backpropogation with batch gradient descent.

--perturb : experimental method for adjusting values
			by .0001 before running backprop, to 
			demonstrate relative insensitivity to
			small changes.

--epochs <int> : specify the number of training epochs to run
				 during training. Defaults to 1.

--eta <float> : specify an initial learning rate for
				backpropogation. Defaults to .01.

--hidden <int> : specifies the number of units to use in
				 hidden layer.

--linear : use a linear transform on the output
		   (e.g. S(s) = s). Otherwise will use tanh(s).

--one : specifies to use a single data point for the
		run, with X = [1,1] and y = 1. To demonstrate
		correctness of backpropogation for a single example.

When the network has finished running, it will display
the two-dimensional plot of its predictions on the supplied
digits data in terms of the extracted features, with ones
in blue and all other digits in red.
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

	print(ones)
	print(not_ones)
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
				 learning_rate=.0001,
				 epochs=1,
				 activation_fn=sigmoid,
				 output_fn = np.tanh,
				 variable_learning_rate=False,
				 weight_decay=False,
				 early_stopping=False):

		self.epochs = epochs
		self.learning_rate = learning_rate
		self.activation_fn = activation_fn
		self.output_fn = output_fn
		self.size_in = size_in
		self.size_hidden = size_hidden
		self.variable_learning_rate = variable_learning_rate
		self.weight_decay = weight_decay
		self.early_stopping = early_stopping

		if self.early_stopping and (self.variable_learning_rate or self.weight_decay):
			raise ValueError("Incompatible regularization methods for backpropogation.")

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

		#Set holdout variables for early stopping
		if self.early_stopping:
			best_Eval = float('inf')
			holdout_input_hidden_weights = copy(self.input_hidden_weights)
			holdout_hidden_output_weights = copy(self.hidden_output_weights)
			validation_errors = []

		#Backpropogation with batch gradient descent
		E_in = []
		D = np.concatenate(([[y] for y in Y], X), axis=1)
		for epoch in range(self.epochs):
			np.random.shuffle(D)

			#Get in-sample error before update
			results = []
			for example in D:
				res, node_signals, node_outputs = self.__feedforward(example[1:], True)
				results.append(res)
			orig_E_total = .25 * sum([
								(results[i] - D[i,0])**2 
								for i in range(len(results))
							]) * (1.0 / float(len(results)))

			#Extra weight decay penalty on in-sample error
			if self.weight_decay:
				total_weights = np.sum(self.input_hidden_weights) + sum(self.hidden_output_weights)
				orig_E_total += (0.01 / float(len(X))**2) * total_weights
			if orig_E_total == 0.0:
				return


			#Beginning of batch gradient descent for single iteration
			gradient = []
			gradient.append(np.array([[0.0 for j in range(self.size_hidden)] for i in range(self.size_in + 1)]))
			gradient.append(np.array([[0.0 for j in range(1)] for i in range(self.size_hidden + 1)]))
			deltas = copy(gradient)


			#If we're using early stopping, partition the dataset
			if self.early_stopping:
				training = D[:250]
				validation = D[250:]
			else:
				training = D


			#Begin actual training process with backprop
			for example in training:

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

				#Adjust gradient if we're using weight decay
				if self.weight_decay:
					gradient[0] = gradient[0] + (0.02 / float(len(X))**2) * self.input_hidden_weights
					gradient[1] = gradient[1] + (0.02 / float(len(X))**2) * self.hidden_output_weights

			#Update weights with gradient
			orig_input_hidden_weights = copy(self.input_hidden_weights)
			orig_hidden_output_weights = copy(self.hidden_output_weights)
			self.input_hidden_weights = self.input_hidden_weights - (self.learning_rate * gradient[0])
			self.hidden_output_weights = self.hidden_output_weights - (self.learning_rate * gradient[1])


			#If we're using variable learning rates, check how to adjust
			if self.variable_learning_rate:

				#Calculate E_total with updated weights
				test_results = []
				for example in training:
					res = self.__feedforward(example[1:], False)
					test_results.append(res)

				test_E_total = .25 * (1.0 / float(len(test_results))) * sum(
									[(test_results[i] - training[i,0])**2 for i in range(len(results))
								])
				if self.weight_decay:
					total_weights = np.sum(self.input_hidden_weights**2) + sum(self.hidden_output_weights**2)
					test_E_total += (0.01 / float(len(X))**2) * total_weights

				#Check whether to accept or reject new weights, adjust learning rate appropriately
				alpha = 1.1
				beta = .9
				if test_E_total < orig_E_total:
					self.learning_rate = self.learning_rate * alpha
					E_in.append(test_E_total)
				else:
					self.learning_rate = self.learning_rate * beta
					self.input_hidden_weights = orig_input_hidden_weights
					self.hidden_output_weights = orig_hidden_output_weights
					E_in.append(orig_E_total)


			#If we're using early stopping, determine whether or not to reassign holdout weights
			elif self.early_stopping:

				#Calculate validation error 
				validation_results = []
				for example in validation:
					validation_results.append(self.__feedforward(example[1:], False))

				validation_E_total = .25 * (1.0 / float(len(validation_results))) * sum(
					[(validation_results[i] - validation[i,0])**2 for i in range(len(validation))
				])
				validation_errors.append(validation_E_total)

				#Reassign holdout weights, best validation error
				if validation_E_total < best_Eval:
					best_Eval = validation_E_total
					holdout_input_hidden_weights = copy(self.input_hidden_weights)
					holdout_hidden_output_weights = copy(self.hidden_output_weights)
			else:
				pass

			#Only do this if we're using the '--ones' test case
			if len(X) == 1:
				print("Gradient: ")
				print(gradient[0], gradient[1])
				print("Weights: ")
				print(self.input_hidden_weights)
				print(self.hidden_output_weights)

		#For early stopping, set holdout weights determined via cross-validation
		if self.early_stopping:
			self.input_hidden_weights = holdout_input_hidden_weights
			self.hidden_output_weights = holdout_hidden_output_weights
			print("Optimal cross-validation error: ", best_Eval)

			#Plot cross-validation error
			plt.plot(range(self.epochs), validation_errors, color='b')
			plt.show()
			plt.close()			

		#Plot the in-sample versus training iteration if using variable learning rates
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
		learning_rate = .0001

	if '--epochs' in sys.argv:
		epochs = int(sys.argv[sys.argv.index('--epochs') + 1])
	else:
		epochs = 1

	if '--variable' in sys.argv:
		variable_learning_rate = True
	else:
		variable_learning_rate = False

	if '--decay' in sys.argv:
		weight_decay = True
	else:
		weight_decay = False

	if '--early-stopping' in sys.argv:
		early_stopping = True
	else:
		early_stopping = False

	#Train neural net
	net = NeuralNet(
				size_in=size_in,
				size_hidden=size_hidden, 
				activation_fn=sigmoid,
				output_fn=output_fn,
				epochs=epochs,
				learning_rate=learning_rate,
				variable_learning_rate=variable_learning_rate,
				weight_decay=weight_decay,
				early_stopping=early_stopping
			)
	net.train(D[:300,1:], D[:300,0])

	#Predict test set, get test error and plot
	predictions = net.predict(D[300:,1:])
	test_err = float(len(
			[
				predictions[i] 
				for i in range(len(predictions))
				if not predictions[i] == D[300 + i,0]
			]
		)) / float(len(predictions))
	print("Test Error: ", test_err)
	
	__plot_decision_boundary(D[300:], predictions)



if __name__ == '__main__':
	sys.exit(main())