#!/usr/bin/env python3
'''
Author: Spencer Norris
File: main.py
Description: implements a full machine learning pipeline
for the pseudo-inverse linear regression problem,
including a non-linear transform of the data,
regularization and cross-validation 
'''

import matplotlib.pyplot as plt
import numpy as np
import math
import sys


lamb = None
cross_val = False


def __linear_regression_weights(X, y, lamb):
	'''
	Analytically computes the weights for linear regression with regularization.
	Expects data matrix X, column matrix y and regularization constant lamb.
	'''
	def __multiply_matrices(A, B):
		'''
		Returns the results of performing matrix multiplication
		on two numpy arrays A and B.
		'''
		return np.array([[np.dot(a,b) for b in B.T] for a in A])


	#Use inverse to find analytical solution for weights
	inverse = np.linalg.inv(np.matmul(X.T, X) + lamb * np.identity(len(X.T)))
	pseudo = np.matmul(inverse, X.T)
	return np.matmul(pseudo, y)


def __get_cv_err(lamb, X, Y):
	def __multiply_matrices(A, B):
		'''
		Returns the results of performing matrix multiplication
		on two numpy arrays A and B.
		'''
		return np.array([[np.dot(a,b) for b in B.T] for a in A])

	inverse = np.linalg.inv(np.matmul(X.T, X) + lamb * np.identity(len(X.T)))
	pseudo = np.matmul(np.matmul(X, inverse), X.T)
	predictions = np.matmul(pseudo,Y)
	total = 0.0
	for i in range(len(predictions)):
		total += ( (predictions[i] - Y.T[i]) / (1 - pseudo[i][i]) )**2
	return total / len(predictions)


def __build_polynomial_transformed_matrix(D):
	'''
	Perform an 8th-order orthogonal polynomial transform 
	on both datasets using the Legendre transform.
	'''
	def __legendre_transform(x, k):
		'''
		Performs a k-th order Legendre transform on the input x.
		'''
		#Base cases
		if k == 0:
			return 1
		if k == 1:
			return x

		term_one = ((2*float(k) - 1) / float(k) ) * x * __legendre_transform(x, k-1)
		term_two = ((float(k) - 1) / float(k)) * __legendre_transform(x, k-2)
		return term_one - term_two


	#Builds the degree-8 polynomial transform row
	def __build_transformed_row(x):
		res = []
		res.append(1)
		assert(len(x) == 2)
		for num_terms in range(2,10): #require one zero of degree 0, two terms deg 1...
			for i in range(0, num_terms):
				res.append(__legendre_transform(x[0], num_terms - 1 - i) * __legendre_transform(x[1], i))
		assert(len(res) == 45)
		return res


	final = []
	for x in D:
		final.append(__build_transformed_row(x))
	return np.array(final)


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


def main():
	global lamb
	global cross_val
	weights = None

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
	X = [[2* (x[0] - symm_min) / (symm_max - symm_min) -1, x[1]] for x in X]


	#Re-attach Y to X, assign to D, Shuffle dataset and partition
	D = np.concatenate(([[y] for y in Y], X), axis=1)
	np.random.shuffle(D)

	#Get the polynomial transformed Z matrix for training, append Y
	Z = __build_polynomial_transformed_matrix(D[:, 1:])
	print("Transformed Z matrix shape: ", Z.shape)
	Z = np.concatenate(([[y] for y in D[:,0]], Z), axis=1)

	#Partition Z into training and test sets
	training = Z[:300]
	test = Z[300:]

	#Perform cross-validation?
	if cross_val:
		best_weights = None
		best_err = float('inf')
		best_lamb = None

		#iterate over lambda selections, test
		cross_val_errors = [] # record errors for lambdas
		test_errors = []
		for test_lamb in np.arange(0.0, 2.0, .01):

			#Get total crossval error for lambda
			cv_err = __get_cv_err(test_lamb, Z[:,1:], Z[:,0])
			cross_val_errors.append(cv_err)

			#Get test error
			test_weights = __linear_regression_weights(
					training[:,1:],
					np.array([[y] for y in training[:,0]]),
					test_lamb
				).T[0]
			test_predictions = []
			for point in test[:,1:]:
				test_predictions.append(np.dot(test_weights, point))
			test_predictions = np.sign(np.array(test_predictions))
			test_err = float(len(
				[
					test_predictions[i] 
					for i in range(len(test_predictions))
					if not test_predictions[i] == test[i][0]
				]
			)) / float(len(test_predictions))
			test_errors.append(test_err)

			#Update optimal crossval error, lamb
			if cross_val_errors[len(cross_val_errors) - 1] < best_err:
				best_err = cross_val_errors[len(cross_val_errors) - 1]
				best_lamb = test_lamb

		print("Cross-validation error: ", best_err)
		print("Optimal lambda: ", best_lamb)
		weights = __linear_regression_weights(
					training[:,1:],
					np.array([[y] for y in training[:,0]]),
					best_lamb
				).T[0]

		#Plot crossval, test errors against lambda
		plt.plot(np.arange(0.0, 2.0, .01), cross_val_errors, color='b')
		plt.plot(np.arange(0.0, 2.0, .01), test_errors, color='r')
		plt.show()
		plt.close()


	else:
		#Get regularized regression weights using appropriate lambda
		weights = __linear_regression_weights(
					training[:,1:],
					np.array([[y] for y in training[:,0]]),
					lamb if lamb is not None else 0
				).T[0]


	#Classify test points
	test_predictions = []
	for point in test[:,1:]:
		test_predictions.append(np.dot(weights, point))
	test_predictions = np.sign(np.array(test_predictions))

	#Get test error
	test_err = float(len(
		[
			test_predictions[i] 
			for i in range(len(test_predictions))
			if not test_predictions[i] == test[i][0]
		]
	)) / float(len(test_predictions))
	print("Test Error: ", test_err)

	#Separate test points into +1s, -1s
	ones = np.array([ 
		D[300 + i, 1:] 
		for i in range(len(D[300:]))
		if test_predictions[i] == 1
	])
	not_ones = np.array([ 
		D[300 + i, 1:] 
		for i in range(len(D[300:]))
		if not test_predictions[i] == 1
	])

	#Plot points using original 2-D coordinates
	fig, ax = plt.subplots()
	ax.scatter(ones[:,0], ones[:,1], color='b', alpha=.2)
	ax.scatter(not_ones[:,0], not_ones[:,1], color='r', alpha=.2)
	ax.set_xlabel("Anti-Symmetry")
	ax.set_ylabel("Avg. Intensity")
	plt.show()
	plt.close()

	return 0

if __name__ == '__main__':
	if '--use-lamb' in sys.argv:
		lamb = float(sys.argv[sys.argv.index('--use-lamb') + 1])
	if '--cross-val' in sys.argv:
		cross_val = True
	sys.exit(main())