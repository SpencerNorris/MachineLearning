#!/usr/bin/env python3
'''
Author: Spencer Norris
File: separable_svm.py
Description: implements a quadratic programming
solution for obtaining the optimal separating
hyperplane of a support vector machine
use a two-point dataset consisting of 
x_1 = [1,0], y_1 = 1, x_2 = [-1,0] y_2 = -1.

Use the following flag in the command line
to configure the behavior of the script:

--z-space : performs a transformation of data points
'''

import matplotlib.pyplot as plt
from cvxopt import solvers
import cvxopt
import numpy as np
import sys


def Z_transform(D):
	'''
	Performs the following transformation:
	[x1, x2] --> [x1^3 - x2, x1*x2]
	'''
	return np.array([
			[X[0], X[1]**3 - X[2], X[1]*X[2]]
			for X in D
		])


class SVM():
	def __init__(self):
		self.bias = 0
		self.weights = None
		self.dimensions = None

	def train(self,D):
		'''
		Expects Y values of each example
		in first position of row, X values
		in positions 1 through d.
		'''
		X = D[:, 1:]
		Y = D[:,0]

		#Build Q matrix
		Q = np.identity(len(X.T))
		Q = np.concatenate((np.array([[0] for i in range(len(Q))]), Q), axis=1)
		Q = np.concatenate((np.zeros((1, len(X.T) + 1)), Q), axis=0)
		Q = cvxopt.matrix(Q, tc='d')

		#Build A matrix
		A = np.array([Y[i]*X[i] for i in range(len(D))])
		A = -1 * np.concatenate( (np.array([[y] for y in Y]), A), axis=1)
		A = cvxopt.matrix(A, tc='d')
		print(A)

		p = np.zeros((len(X.T) + 1, 1))
		p = cvxopt.matrix(p, tc='d')

		c = -1 * np.ones((len(D)))
		c = cvxopt.matrix(c, tc='d')

		solution = solvers.qp(Q,p,A,c)['x']

		self.bias = solution[0]
		self.weights = np.array(solution)[1:].T[0]
		self.dimensions = len(self.weights)

		print("Bias: ", self.bias)
		print("Weights: ", self.weights)


	def classify(self, X):
		return np.sign(np.dot(self.weights, X) + self.bias)


	def linear_plotpoint(self,x):
		'''
		Returns the linear decision boundary at that particular point
		for visualizing the boundary. Accepts x coordinate, returns y.
		Only works when training in 2-D!!!
		'''
		return -1*(self.weights[0] / self.weights[1]) * x - self.bias

	def project_plotpoint(self,x):
		'''
		Returns the Z-space decision boundary projected into X-space.
		Accepts the x coordinate, returns y coordinate.
		Again, only works when training data is 2-D!
		'''
		return ((-self.bias) - (self.weights[0] * x**3)) / (-self.weights[0] + self.weights[1]*x)



def main():
	D = np.array([
			[1, 1, 0],
			[-1, -1, 0]
		])

	linear_svm = SVM()
	linear_svm.train(D)

	#Create plot of X-space SVM
	plt.scatter(D[:,1], D[:,2])
	if linear_svm.weights[1] == 0.0:
		plt.axvline(x=0.0)
	else:
		plt.plot(np.arange(-1,1), [linear_svm.plotpoint(x) for x in np.arange(-1,1)])
	plt.title("X-Space SVM")
	plt.show()
	plt.close()

	#Perform Z-space transform, plot decision boundary
	if '--z-space' in sys.argv:
		D_z = Z_transform(D)
		non_linear_svm = SVM()
		non_linear_svm.train(D_z)
		plt.scatter(D_z[:,1], D_z[:,2])
		if non_linear_svm.weights[1] == 0.0:
			plt.axvline(x=0.0)
		else:
			plt.plot(np.arange(-1,1), [non_linear_svm.plotpoint(x) for x in np.arange(-1,1)])
		plt.title("Z-Space SVM")
		plt.show()
		plt.close()


		#Place both SVMs in X-space
		#Plot linear SVM
		plt.scatter(D[:,1], D[:,2])
		if linear_svm.weights[1] == 0.0:
			plt.axvline(x=0.0)
		else:
			plt.plot(np.arange(-1,1),
				[linear_svm.plotpoint(x)
				for x in np.arange(-1,1)])

		#Project Z-space SVM
		plt.plot(np.arange(-1,1,.01),
			[non_linear_svm.project_plotpoint(x)
			for x in np.arange(-1,1,.01)],
			'#d95f0e')
		plt.title("X-Space and Projected Z-Space SVMs")
		plt.show()
		plt.close()



if __name__ == '__main__':
	sys.exit(main())