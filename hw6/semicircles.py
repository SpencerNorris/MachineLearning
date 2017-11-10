#!/usr/bin/env python3
'''
Author: Spencer Norris
File: semicircles.py
Description: solution to problem 3.1 in Learning from Data
by Abu-Mostafa, Magdon-Ismail and Lin.
'''

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from random import uniform, shuffle
from copy import copy
import numpy as np
import random
import math
import sys


#Just a big copy-pasta from HW1, with slight modifications
class Perceptron():
	def __init__(self, dims=2):
		self.weights = np.random.uniform(.01, .05, dims + 1)

	def __update_weights(self, example, true_label):
		self.weights = self.weights + int(true_label)*np.append(example, [1])

	def train(self, data):
		num_examples = len(data)
		updates = 0

		#Wait until convergence
		while True:

			#Iterate over all examples
			#If we misclassify, restart
			#Otherwise, count correct classifications
			np.random.shuffle(data)
			correct_classifications = 0
			for row in data:
				true_label = row[-1]
				label = self.classify(row[:-1])
				if not int(label) == int(true_label):
					self.__update_weights(row[:-1], true_label)
					updates += 1
					break
				else:
					correct_classifications += 1

			#If we classified every example correctly, we've converged
			if correct_classifications == num_examples:
				break
		return updates

	def classify(self, example):
		return 1 if np.dot(self.weights, np.append(example,[1.0])) > 0 else -1



def __linear_regression_weights(X, y):
	'''
	Analytically computes the weights for linear regression.
	'''
	def __multiply_matrices(A, B):
		'''
		Returns the results of performing matrix multiplication
		on two numpy arrays A and B.
		'''
		return np.array([[np.dot(a,b) for b in B.T] for a in A])


	#Reshape y, append bias variable to X
	bias = np.ones((len(X),1))
	X = np.append(X,bias,1)

	#Calculate the pseudoinverse of X
	inverse = np.linalg.inv(__multiply_matrices(X.T, X))
	pseudo = __multiply_matrices(inverse, X.T)
	return __multiply_matrices(pseudo, y)


def __top_semicircle(radius,thickness):
	'''
	Generates a random 2-D point lying in the top semi-circle.
	'''
	scale = radius + random.uniform(0,thickness)
	angle = math.radians(random.uniform(0,180))
	x = math.cos(angle) * scale
	y = math.sin(angle) * scale
	return [x,y]



def __bottom_semicircle(radius,thickness,separation):
	'''
	Generates a random 2-D point lying in the bottom semi-circle.
	'''
	scale = radius + random.uniform(0,thickness)
	angle = math.radians(random.uniform(180,360))
	x = (math.cos(angle) * scale) + radius + .5*thickness
	y = (math.sin(angle) * scale) - separation
	return [x,y]


def main():
	#generate random points in semicircles

	top_data = [__top_semicircle(radius,thickness) for i in range(1000)]
	top_x = list(zip(*top_data))[0]
	top_y = list(zip(*top_data))[1]

	bottom_data = [__bottom_semicircle(radius,thickness,separation) for i in range(1000)]
	bottom_x = list(zip(*bottom_data))[0]
	bottom_y = list(zip(*bottom_data))[1]


	#Set up perceptron, train
	perceptron = Perceptron()
	for i in range(len(top_data)):
		top_data[i].append(1)
	for i in range(len(bottom_data)):
		bottom_data[i].append(-1)
	training_data = np.concatenate((np.array(bottom_data), np.array(top_data)))
	training_data = np.array(training_data)
	iterations = perceptron.train(training_data)
	print("Training Iterations: ", iterations)

	#Graph data
	fig,ax = plt.subplots()
	ax.scatter(top_x, top_y, color='red')
	ax.scatter(bottom_x, bottom_y, color='blue')

	#Graph perceptron
	perceptron_weights = perceptron.weights
	p_w_0 = perceptron_weights[-1]
	p_w_1 = perceptron_weights[0]
	p_w_2 = perceptron_weights[1]
	perceptron_boundary = lambda x1 : -1*(p_w_1/p_w_2)*x1 - (p_w_0/p_w_2)
	perceptron_line =  mlines.Line2D(
					[-20,50], 
					[perceptron_boundary(-20), perceptron_boundary(50)], 
					color='green')
	ax.add_line(perceptron_line)
	plt.show()
	fig.clear()

	#Get linear regression weights, graph
	reg_data_x = np.concatenate((np.array([[x] for x in bottom_x]), np.array([[x] for x in top_x])), axis=0)
	reg_data_y = np.concatenate((np.array([[y] for y in bottom_y]), np.array([[y] for y in top_y])), axis=0)
	fig,ax = plt.subplots()
	ax.scatter(top_x, top_y, color='red')
	ax.scatter(bottom_x, bottom_y, color='blue')
	reg_weights = __linear_regression_weights(reg_data_x[:300], reg_data_y[:300])
	r_w_0 = reg_weights[-1]
	r_w_1 = reg_weights[0]
	r_w_2 = reg_weights[1]
	reg_boundary = lambda x1 : -1*(r_w_1/r_w_2)*x1 - (r_w_0/r_w_2)
	reg_line =  mlines.Line2D(
					[-20,50], 
					[reg_boundary(-20), reg_boundary(50)], 
					color='green')
	ax.add_line(reg_line)
	plt.show()


	return 0

if __name__ == '__main__':
	radius = float(sys.argv[1])
	thickness = float(sys.argv[2])
	separation = float(sys.argv[3])
	sys.exit(main())