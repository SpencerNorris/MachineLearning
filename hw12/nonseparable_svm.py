#!/usr/bin/env python3
'''
Author: Spencer Norris
File: separable_svm.py
Description:

'''

import matplotlib.pyplot as plt
from cvxopt import solvers
import cvxopt
import numpy as np
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


def polynomial_kernel(x,y):
	return (1 + np.dot(x,y))**8


class KSoftMarginSVM():
	'''
	Implementation of a soft-margin support vector machine
	using a provided kernel function.
	'''
	def __init__(self, kernel_fn=polynomial_kernel, reg_param=1000):
		self.bias = 0.0
		self.alphas = None
		self.weights = None
		self.support_vectors = None
		self.kernel_fn = kernel_fn
		self.C = reg_param

	def train(self,D):
		'''
		Expects Y values of each example
		in first position of row, X values
		in positions 1 through d.

		Returns upper bound of cross-validation
		error in terms of the number of support vectors.
		'''
		X = D[:, 1:]
		Y = D[:,0]

		#Build kernel matrix
		K = np.array([
				[self.kernel_fn(X[i], X[j]) for j in range(len(X))]
				for i in range(len(X))
			])

		#Build Q matrix
		Q = np.array([
				[Y[i]*Y[j]*K[i,j] for j in range(len(X))]
				for i in range(len(X))
			])
		Q = cvxopt.matrix(Q, tc='d')

		#Build A matrix (negatives for > 0, positives for < C)
		A = np.concatenate((-np.identity(len(X)), np.identity(len(X))))
		A = cvxopt.matrix(A, tc='d')

		#Build p vector
		p = -1 * np.ones((len(X)))
		p = cvxopt.matrix(p, tc='d')

		#Build C vector (0 for > 0, C for < C)
		c = np.concatenate( (np.zeros((len(X))), self.C * np.ones(len(X))) )
		c = cvxopt.matrix(c, tc='d')

		#Use Y.T dot alphas for equality to 0
		matrix_y = cvxopt.matrix(Y, (1,len(X)), tc='d')
		b = cvxopt.matrix(0.0)

		#Get alphas using convex optimization, select support vectors
		support_vectors = []
		support_alphas = []
		alphas = np.ravel(solvers.qp(Q,p,A,c,matrix_y,b)['x'].T)
		for i in range(len(alphas)):
			if alphas[i] > 1e-5:
				support_vectors.append(D[i])
				support_alphas.append(alphas[i])
		self.alphas = support_alphas
		self.support_vectors = support_vectors
		
		#Recover weights from alphas
		self.weights = sum([self.support_vectors[i][0]*self.alphas[i]*self.support_vectors[i][1:]
							 for i in range(len(self.alphas))])

		#Estimate bias in terms of support vectors
		self.bias = 0.0
		for vec in self.support_vectors:
			self.bias += vec[0] - np.dot(self.weights, vec[1:])
		self.bias /= float(len(self.support_vectors))

		return float(len(self.support_vectors)) / float(len(D))


	def classify(self, X):
		s = 0.0
		for alpha, vec in zip(self.alphas, self.support_vectors):
			s += alpha * vec[0] * self.kernel_fn(X, vec[1:])
		return np.sign(s + self.bias)


def main():
	solvers.options['show_progress'] = False
	if '--C' in sys.argv:
		selected_C = float(sys.argv[sys.argv.index('--C') + 1])
	else:
		selected_C = None

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

	#Perform cross-validation to select optimal C
	if selected_C is None:
		D_train = D[:300]
		selected_C = 0.0
		best_E_cv = float('inf')
		crossval_errors = []
		for C in np.arange(0.001,5.001,.5):
			print("Testing C =", C)
			#Get crossval error
			E_cv = 0.0
			for i in range(len(D_train)):
				D_cv = np.delete(D_train, (i), axis=0)
				svm = KSoftMarginSVM(
					kernel_fn=polynomial_kernel,
					reg_param=C)
				E_cv += svm.train(D_cv)
			E_cv /= len(D_train)
			crossval_errors.append(E_cv)

			#Check for optimality of C
			if E_cv < best_E_cv:
				best_E_cv = E_cv
				selected_C = C

		print("Optimal C: ", selected_C)

		#Plot crossval error over iterations
		plt.plot(np.arange(0.001,5.001,.5), crossval_errors, color='b')
		plt.show()
		plt.close()

	#Instantiate SVM with final selection of C
	svm = KSoftMarginSVM(
		kernel_fn=polynomial_kernel,
		reg_param=selected_C)
	svm.train(D[:300])


	#Classify test data, plot predictions
	D_test = D[300:]
	predictions = []
	for example in D_test:
		predictions.append(svm.classify(example[1:]))
	E_test = len([True for i in range(len(D_test))
				  if not D_test[i,0] == predictions[i]
			 ]) / len(D_test)

	print("E_test: ", E_test)
	__plot_decision_boundary(D_test, predictions)


if __name__ == '__main__':
	sys.exit(main())