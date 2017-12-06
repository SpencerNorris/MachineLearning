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

from copy import copy
import numpy as np
import math
import sys



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
				 learning_rate=.01,
				 num_epochs=1,
				 activation_fn=sigmoid,
				 output_fn = np.tanh,
				 update_method='backprop'):

		self.num_epochs = num_epochs
		self.learning_rate = learning_rate
		self.activation_fn = activation_fn
		self.output_fn = output_fn
		self.update_method = update_method
		self.size_in = size_in
		self.size_hidden = size_hidden

		#Input-to-hidden weights (Include extra weight for the bias terms)
		self.input_hidden_weights = np.array([[.25 for j in range(size_hidden)] for i in range(size_in + 1)])
		self.hidden_output_weights = np.array([[.25 for j in range(1)] for i in range(size_hidden + 1)])


	def __perturb():
		'''
		
		'''
		pass


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
		#Check if we're just using numerical perturbations
		if not self.update_method == 'backprop':
			self.__perturb()
			return


		#Otherwise we're going to use backpropogation with batch gradient descent
		for epoch in range(self.num_epochs):
			results = []
			for x in X:
				res, node_signals, node_outputs = self.__feedforward(x, True)
				results.append(res)

			#Calculate E_total
			E_total = .25 * (1.0 / float(len(results))) * sum([(results[i] - Y[i])**2 for i in range(len(results))])
			if E_total == 0.0:
				return

			updated_input_hidden_weights = copy(self.input_hidden_weights)
			updated_hidden_output_weights = copy(self.hidden_output_weights)

			gradient = []
			gradient.append(np.array([[0.0 for j in range(self.size_hidden)] for i in range(self.size_in + 1)]))
			gradient.append(np.array([[0.0 for j in range(1)] for i in range(self.size_hidden + 1)]))
			deltas = copy(gradient)

			for i in range(len(X)):
				#Feedforward
				x = X[i]
				res, node_signals, node_outputs = self.__feedforward(x, True)

				#Get deltas for example
				__output_deltas = np.array([0.0 for i in range(1)])
				__hidden_deltas = np.array([0.0 for i in range(self.size_hidden)])

				#Deltas for output
				if self.output_fn is np.tanh:
					output_derivative = lambda x: 1 - node_outputs[2]**2
				else:
					output_derivative = lambda x: 1
				sig_derivative = lambda x: sigmoid(x) * (1 - sigmoid(x))
				__output_deltas = 2 * (node_outputs[2] - Y[i]) * output_derivative(node_signals[2])

				#Deltas for hidden layer
				sig_deriv_transform = np.array([ #Applies the derivative of the sigmoid function element-wise
										[sig_derivative(node_signals[1][i])] 
										for i in range(len(node_signals[1]))
									])
				__hidden_deltas = sig_deriv_transform * self.hidden_output_weights[1:] * __output_deltas

				__hidden_deltas = __hidden_deltas / float(len(X))
				__output_deltas = __output_deltas / float(len(X))
				
				#Update the gradient with deltas
				gradient[0] = gradient[0] + np.outer(node_outputs[0].T[0], __hidden_deltas.T[0])
				gradient[1] = gradient[1] + np.outer(node_outputs[1].T[0], __output_deltas)

				self.input_hidden_weights = self.input_hidden_weights - self.learning_rate * gradient[0]
				self.hidden_output_weights = self.hidden_output_weights - self.learning_rate * gradient[1]

				if len(X) == 1:
					print("Gradient: ")
					print(gradient[0], gradient[1])
					print("Weights: ")
					print(self.input_hidden_weights)
					print(self.hidden_output_weights)

	def predict(self, X):
		return np.sign(self.__feedforward(X, False))


def main():
	if '--one' in sys.argv:
		D = np.array([[1.0,1.0,1.0]])
		X = np.array(D[:,:2])
		Y = np.array(D[:,2])
		num_epochs = 1
	else:
		#LOAD IN DATASET
		num_epochs = 2 * 10**6

	size_in = len(X.T)

	if '--hidden' in sys.argv:
		size_hidden = int(sys.argv[sys.argv.index('--hidden') + 1])
	else:
		size_hidden = 2

	if '--linear' in sys.argv:
		output_fn = linear
	else:
		output_fn = np.tanh

	if '--perturb' in sys.argv:
		update_method = 'perturb'
	else:
		update_method = 'backprop'

	if '--eta' in sys.argv:
		learning_rate = float(sys.argv[sys.argv.index('--eta') + 1])
	else:
		eta = .0001

	net = NeuralNet(
				size_in=size_in,
				size_hidden=size_hidden, 
				learning_rate=eta, 
				activation_fn=sigmoid,
				output_fn=output_fn,
				update_method=update_method
			)
	net.train(X, Y)

if __name__ == '__main__':
	sys.exit(main())