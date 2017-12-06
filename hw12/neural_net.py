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
	if x == 0:
		return .5
	x = np.clip(-1000,1000,x)
	return 1 / (1 + math.exp(-x))

def linear(x):
	return x

class NeuralNet():
	def __init__(self, learning_rate=.0001, 
				 activation_fn=sigmoid,
				 output_fn = np.tanh,
				 update_method='backprop'):

		self.learning_rate = learning_rate
		self.activation_fn = activation_fn
		self.output_fn = output_fn
		self.update_method = update_method

		#Input-to-hidden weights (Include extra weight for the bias terms)
		self.input_hidden_weights = np.array([[.25 for j in range(2)] for i in range(3)])
		self.hidden_output_weights = np.array([[.25 for j in range(1)] for i in range(3)])


	def __backprop(self, xin, results, true_vals, node_signals, node_outputs):
		'''
		Update weights in network according to error produced during training.
		'''
		updated_input_hidden_weights = copy(self.input_hidden_weights)
		updated_hidden_output_weights = copy(self.hidden_output_weights)

		#Calculate E_total
		E_total = .25 * sum([(results[i] - true_vals[i])**2 for i in range(len(results))])
		if E_total == 0.0:
			return


		#Update hidden-to-output weights
		gradient = []
		gradient.append([])
		gradient.append([])

		N = len(results)
		dE_total_over_dOut = -(1.0 / (2.0 * N)) * sum([true_vals[i] - results[i] for i in range(len(results))])
		if self.output_fn is np.tanh:
			dOut_over_dNet = 1 - np.tanh(node_signals[2])**2
		else:
			dOut_over_dNet = 1
		for i in range(len(self.hidden_output_weights)):
			dE_total_over_dw = dE_total_over_dOut * dOut_over_dNet * node_outputs[1][i]
			gradient[1].append(dE_total_over_dw)
			updated_hidden_output_weights[i][0] = self.hidden_output_weights[i][0] - self.learning_rate * dE_total_over_dw

		#Update input-to-hidden weights using hidden deltas
		xin = np.append([1], xin)
		for i in range(len(self.input_hidden_weights)):
			for j in range(len(self.input_hidden_weights[i])):
				dE_total_over_dOut_hidden = dE_total_over_dOut * self.hidden_output_weights.T[0][j]
				dOut_over_dNet_hidden = node_outputs[1].T[0][j] * (1 - node_signals[1].T[0][j])
				dE_total_over_dw = dE_total_over_dOut_hidden * dOut_over_dNet_hidden * xin[i]
				gradient[0].append(dE_total_over_dw)
				updated_input_hidden_weights[i][j] = self.input_hidden_weights[i][j] - self.learning_rate * dE_total_over_dw


		self.input_hidden_weights = updated_input_hidden_weights
		self.hidden_output_weights = updated_hidden_output_weights
		print(gradient)


	def __peturb():
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
		#Randomly select a holdout point to use for backprop
		index = np.random.randint(low=0, high=len(X))
		results = []
		holdout_signals = None
		holdout_outputs = None
		for i in range(len(X)):
			res, node_signals, node_outputs = self.__feedforward(X[i], True)
			if i == index:
				holdout_signals = node_signals
				holdout_outputs = node_outputs
			results.append(res)

		print("feedforward result: ", results)

		#Perform backpropogation with full gradient descent
		if self.update_method == 'backprop':
			self.__backprop(X[index], results, Y, holdout_signals, holdout_outputs)
		else:
			self.__perturb()

	def predict(self, X):
		return np.sign(self.__feedforward(X, False))


def main():
	if '--one' in sys.argv:
		D = np.array([[1.0,1.0,1.0]])
		X = np.array(D[:,:2])
		Y = np.array(D[:,2])

	if '--linear' in sys.argv:
		output_fn = linear
	else:
		output_fn = np.tanh

	if '--perturb' in sys.argv:
		update_method = 'perturb'
	else:
		update_method = 'backprop'

	if '--eta' in sys.argv:
		learning_rate = sys.argv[sys.argv.index('--eta') + 1]
	else:
		eta = .0001

	net = NeuralNet(
				learning_rate=eta, 
				activation_fn=sigmoid,
				output_fn=output_fn,
				update_method=update_method
			)
	net.train(X, Y)

if __name__ == '__main__':
	sys.exit(main())