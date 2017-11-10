#!/usr/bin/env python3

'''

Include '--third' in command line args
to perform a third-order transformation on the
features before fitting the perceptron.
'''


import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import math
import sys


use_third_order = False

def classify(weights, example):
	'''
	Uses linear regression weights to get classification of points.
	'''
	return np.sign(np.dot(weights, np.append(example, [1])))

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

def __get_symmetry_score(x):
	'''
	Takes the sums of the squared errors between the pixel values
	flipped along the X axis and Y axis and mapped onto the other side.
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


def main():
	global use_third_order

	third_order = lambda x: x**3 + x**2 + x

	#Load datasets, filter for only 1 and 5 and change labels to -1,+1 (1 and 5 respectively)
	training = np.genfromtxt('training.txt')
	training = training[np.logical_or(training[:,0] == 1, training[:,0] == 5)]
	for row in training:
		assert row[0] == 1 or row[0] == 5
	training_x = training[:,1:]
	training_y = training[:,0]
	training_y = np.array([1 if y == 1 else -1 for y in training_y])

	#Extract features of dataset
	transformed_training_x = np.zeros((len(training_x), 2))
	for i in range(len(training_x)):
		transformed_training_x[i][0] = __get_symmetry_score(training_x[i])
		transformed_training_x[i][1] = __get_avg_intensity(training_x[i])

		#Perform third-order polynomial transform if called for
		if use_third_order:
			transformed_training_x[i][0] = third_order(transformed_training_x[i][0])
			transformed_training_x[i][1] = third_order(transformed_training_x[i][1])

	#Get regression weights
	weights = __linear_regression_weights(transformed_training_x,np.array([[y] for y in training_y]))
	print('Coefficients: \n', weights)

	#Get in-sample error
	correct_classifications = 0
	for i in range(len(transformed_training_x)):
		row = transformed_training_x[i]
		prediction = classify(weights.T[0], row)
		if training_y[i] == prediction:
			correct_classifications += 1
	print("Training Accuracy (1 - E_in): ", float(correct_classifications / len(training)))

	fig,ax = plt.subplots()

	#Plot training data, class 1
	transformed_training_x_class_1 = np.array(
			[
				transformed_training_x[i]
				for i in range(len(transformed_training_x))
				if training_y[i] == -1.0
			]
		)
	ax.scatter(
		transformed_training_x_class_1[:,0], 
		transformed_training_x_class_1[:,1], 
		color='red',
		alpha=.5
	)

	#Plot training data, class 2
	transformed_training_x_class_2 = np.array(
		[
			transformed_training_x[i]
			for i in range(len(transformed_training_x))
			if training_y[i] == 1.0
		]
	)
	ax.scatter( 
		transformed_training_x_class_2[:,0],
		transformed_training_x_class_2[:,1],
		color='blue',
		alpha=.5
	)

	#Plot perceptron line for training
	w_0 = weights[-1]
	w_1 = weights[0]
	w_2 = weights[1]
	regression_boundary = lambda x1 : -1*(w_1/w_2)*x1 - (w_0/w_2)
	regression_line =  mlines.Line2D(
					[-1000000000,250000000], 
					[regression_boundary(-1000000000), regression_boundary(250000000)], 
					color='green')
	ax.add_line(regression_line)
	plt.show()
	plt.close()

	#Transform test data
	test = np.genfromtxt('test.txt')
	test = test[np.logical_or(test[:,0] == 1, test[:,0] == 5)]
	test_x = test[:,1:]
	test_y = test[:,0]
	test_y = np.array([1 if y == 1 else -1 for y in test_y])
	transformed_test_x = np.zeros((len(test_x), 2))
	for i in range(len(test_x)):
		transformed_test_x[i][0] = __get_symmetry_score(test_x[i])
		transformed_test_x[i][1] = __get_avg_intensity(test_x[i])

		#Perform third-order polynomial transform if called for
		if use_third_order:
			transformed_test_x[i][0] = third_order(transformed_test_x[i][0])
			transformed_test_x[i][1] = third_order(transformed_test_x[i][1])


	#Create another plot for the test data
	fig,ax = plt.subplots()
	transformed_test_x_class_1 = np.array(
			[
				transformed_test_x[i]
				for i in range(len(transformed_test_x))
				if test_y[i] == -1.0
			]
		)
	ax.scatter(
		transformed_test_x_class_1[:,0], 
		transformed_test_x_class_1[:,1], 
		color='red',
		alpha=.5
	)

	#Plot training data, class 2
	transformed_test_x_class_2 = np.array(
		[
			transformed_training_x[i]
			for i in range(len(transformed_test_x))
			if test_y[i] == 1.0
		]
	)
	ax.scatter( 
		transformed_test_x_class_2[:,0],
		transformed_test_x_class_2[:,1],
		color='blue',
		alpha=.5
	)
	regression_line =  mlines.Line2D(
				[-1000000000,250000000], 
				[regression_boundary(-1000000000), regression_boundary(250000000)], 
				color='green')
	ax.add_line(regression_line)
	plt.show()
	plt.close()

	#Determine test error
	correct_classifications = 0
	for i in range(len(transformed_test_x)):
		row = transformed_test_x[i]
		prediction = classify(weights.T[0], row)
		if test_y[i] == prediction:
			correct_classifications += 1
	print("Test Accuracy (1 - E_test): ", float(correct_classifications / len(test)))

	return 0


if __name__ == '__main__':
	if '--third' in sys.argv:
		use_third_order = True
	sys.exit(main())