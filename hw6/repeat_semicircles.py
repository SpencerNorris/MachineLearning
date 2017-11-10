#!/usr/bin/env python3
'''
Author: Spencer Norris
File: semicircles.py
Description: solution to problem 3.2 in Learning from Data
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
	print(y)

	#Calculate the pseudoinverse of X
	inverse = np.linalg.inv(__multiply_matrices(X.T, X))
	pseudo = __multiply_matrices(inverse, X.T)
	return __multiply_matrices(pseudo, y)



def main():
	#generate random points in semicircles
	separations = []
	iteration_values = []

	#Get 10 iteration values for each separation
	for i in range(10):
		for separation in np.arange(.2,5.0,.02):
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

			#Record result
			separations.append(separation)
			iteration_values.append(iterations)


	#Graph results
	fig,ax = plt.subplots()
	ax.scatter(separations, iteration_values, color='blue')
	reg_weights = __linear_regression_weights(np.array([[sep] for sep in separations])[:300],
											 np.array([[it] for it in iteration_values])[:300])
	print(reg_weights)
	r_w_0 = reg_weights[-1]
	r_w_1 = reg_weights[0]
	r_w_2 = reg_weights[1]
	reg_boundary = lambda x1 : (r_w_1)*x1 + (r_w_0)
	reg_line =  mlines.Line2D(
					[-20,50], 
					[reg_boundary(-20), reg_boundary(50)], 
					color='green')
	ax.add_line(reg_line)
	plt.show()

if __name__ == '__main__':
	radius = float(sys.argv[1])
	thickness = float(sys.argv[2])
	sys.exit(main())