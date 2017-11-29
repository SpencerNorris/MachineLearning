#!/usr/bin/env python3
'''
Author: Spencer Norris
File: main.py
Description: 
'''

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import math
import sys


def __plot_decision_boundary(D, test_predictions):
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


def __build_gaussian_kernel_matrix(D, centers, radius):
	'''
	Transforms D into feature space and returns a new
	data matrix Z, where each row of Z corresponds to
	the same row of D and each element of that row
	is the Gaussian of the scaled difference between
	that element and one of the centers determined
	by K-means clustering.
	This new matrix will be N by K+1, where K is the 
	number of centers used (+1 for bias term).
	'''
	def __build_transformed_row(x):
		features = np.array([1 for i in range(len(centers) + 1)])
		for i in range(len(centers)):
			features[i + 1] = np.linalg.norm(x - centers[i])
		features = features / radius
		features = np.power(features, 2)
		features = -.5 * features
		features = np.exp(features)
		features[0] = 1
		return features

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


def __linear_regression_weights(X, y, lamb):
	'''
	Analytically computes the weights for linear regression with regularization.
	Expects data matrix X, column matrix y and regularization constant lamb.
	'''
	#Use inverse to find analytical solution for weights
	inverse = np.linalg.inv(np.matmul(X.T, X) + lamb * np.identity(len(X.T)))
	pseudo = np.matmul(inverse, X.T)
	return np.matmul(pseudo, y)


def __run_rbf_pocket_algorithm(centers, X, Y, runs):
	'''
	Runs the pocket algorithm for a radial basis function network
	using a collection of provided centroids for the clusters.
	'''

	#Initialize K+1 weights
	best_weights = np.random.uniform(0.0, 1.0, len(centers) + 1)

	#Compute optimal weights, error
	best_err = float('inf')
	curr_weights = np.random.uniform(0.0, 1.0, len(centers) + 1)
	for run in range(runs):

		#Classify random test point
		ind = np.random.randint(0, len(X))
		classification = np.sign(np.dot(curr_weights, X[ind]))

		#Check classification
		if not classification == Y[ind]:
			curr_weights = curr_weights + (Y[ind] * X[ind])

			#Compute E_in with updated weights
			num_wrong = 0.0
			for i in range(len(X)):
				if not np.sign(np.dot(curr_weights, X[i])) == Y[i]:
					num_wrong += 1
			err = num_wrong / float(len(Y))

			#Pocket weights?
			if err < best_err:
				best_err = err
				best_weights = np.copy(curr_weights)

	return best_weights, best_err


def __get_linear_cv_err(lamb, X, Y):
	'''
	Use analytical calculation for cross-validation
	error of linear regression.
	'''

	inverse = np.linalg.inv(np.matmul(X.T, X) + lamb * np.identity(len(X.T)))
	pseudo = np.matmul(np.matmul(X, inverse), X.T)
	predictions = np.matmul(pseudo,Y)
	total = 0.0
	for i in range(len(predictions)):
		total += ( (predictions[i] - Y.T[i]) / (1 - pseudo[i][i]) )**2
	return total / len(predictions)


def __get_knn_cv_err(test_k, X, Y):
	'''
	Perform leave one out cross-validation
	and return error for K-nearest neighbors.
	'''
	positives = 0.0
	for i in range(len(X)):
		cv_X = np.delete(X,(i),axis=0)
		cv_Y = np.delete(Y,(i),axis=0)
		model = KNeighborsClassifier(
			n_neighbors=test_k,
			algorithm='auto').fit(cv_X, cv_Y)
		classification = model.predict([X[i]])
		if classification == Y[i]:
			positives += 1
	return 1 - (positives / float(len(X)))



def __get_rbf_cv_err(test_k, X, Y):
	'''
	Runs cross-validation for a Radial Basis Function Network,
	where K-means clustering is used with the provided test_k
	to select the centers for the RBFs and the pocket algorithm
	is used to iteratively update the weights of the network.

	Technically this is incorrect since we compute the centers
	and feature space before removing a point for leave-one-out
	cross-validation; however, the centers will change so little
	that this shouldn't affect the final result significantly.
	'''
	#Select RBF centers using K-means
	kmeans = KMeans(n_clusters=test_k)
	kmeans.fit(X)
	centers = kmeans.cluster_centers_

	#Compute radius
	radius = 2.0 / math.sqrt(float(test_k))

	#Transform X into feature space
	Z = __build_gaussian_kernel_matrix(X, centers, radius)

	num_wrong = 0.0
	for i in range(len(X)):
		cv_X = np.delete(Z,(i),axis=0)
		cv_Y = np.delete(Y,(i),axis=0)

		#Run pocket algorithm
		weights, error = __run_rbf_pocket_algorithm(centers, cv_X, cv_Y, 300)

		#Classify left-out point
		classification = np.sign(np.dot(weights, Z[i]))
		if not classification == Y[i]:
			num_wrong += 1

	return num_wrong / float(len(X))


def main():

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
	X = [[2*((x[0] - symm_min) / (symm_max - symm_min)) - 1, x[1]] for x in X]

	#Re-attach Y to X, assign to D, Shuffle dataset and partition
	D = np.concatenate(([[y] for y in Y], X), axis=1)
	np.random.shuffle(D)


	#============================= Linear regression =================================

	if '--linear' in sys.argv:
		#Get the polynomial transformed Z matrix for training, append Y
		Z = __build_polynomial_transformed_matrix(D[:, 1:])
		Z = np.concatenate(([[y] for y in D[:,0]], Z), axis=1)

		#Partition Z into training and test sets
		training = Z[:300]
		test = Z[300:]

		#iterate over lambda selections, test
		best_weights = None
		best_err = float('inf')
		best_lamb = None
		cross_val_errors = [] # record errors for lambdas
		for test_lamb in np.arange(0.0, 2.0, .01):

			#Get total crossval error for lambda
			cv_err = __get_linear_cv_err(test_lamb, Z[:,1:], Z[:,0])
			cross_val_errors.append(cv_err)

			#Update optimal crossval error, lamb
			if cv_err < best_err:
				best_err = cv_err
				best_lamb = test_lamb
				
		print("Regularized Linear Regression Cross-validation error: ", best_err)
		print("Optimal lambda: ", best_lamb)


		#Use optimal lambda and compute weights
		weights = __linear_regression_weights(
					training[:,1:],
					np.array([[y] for y in training[:,0]]),
					best_lamb
				).T[0]


		#Compute in-sample error
		training_predictions = []
		for point in training[:,1:]:
			training_predictions.append(np.dot(weights, point))
		training_predictions = np.sign(np.array(training_predictions))
		training_err = float(len(
			[
				training_predictions[i] 
				for i in range(len(training_predictions))
				if not training_predictions[i] == training[i][0]
			]
		)) / float(len(training_predictions))
		print("Regularized Linear Regression In-Sample Error: ", training_err)


		#Classify test points, get test error
		test_predictions = []
		for point in test[:,1:]:
			test_predictions.append(np.dot(weights, point))
		test_predictions = np.sign(np.array(test_predictions))
		test_err = float(len(
			[
				test_predictions[i] 
				for i in range(len(test_predictions))
				if not test_predictions[i] == test[i][0]
			]
		)) / float(len(test_predictions))
		print("Test Error: ", test_err)

		#Plot crossval, test errors against lambda
		plt.plot(np.arange(0.0, 2.0, .01), cross_val_errors, color='b')
		plt.show()
		plt.close()

		__plot_decision_boundary(D, test_predictions)


	#============================= K-Nearest Neighbors =================================

	if '--knn' in sys.argv:

		#Partition D into training and test sets
		training = D[:300]
		test = D[300:]

		#iterate over lambda selections, test
		best_err = float('inf')
		best_k = None
		cross_val_errors = [] # record errors for lambdas
		for test_k in range(1, 21):
			#Get crossval error
			cv_err = __get_knn_cv_err(test_k, training[:,1:], training[:,0])
			cross_val_errors.append(cv_err)

			#Update optimal crossval error, k
			if cv_err < best_err:
				best_err = cv_err
				best_k = test_k

		print("KNN Cross-validation error: ", best_err)
		print("Optimal K: ", best_k)


		#Train model using best K
		model = KNeighborsClassifier(
		n_neighbors=best_k,
		algorithm='auto').fit(training[:,1:], training[:,0])


		#Compute in-sample error
		training_predictions = model.predict(training[:,1:])
		training_err = float(len(
			[
				training_predictions[i] 
				for i in range(len(training_predictions))
				if not training_predictions[i] == training[i,0]
			]
		)) / float(len(training_predictions))
		print("K-Nearest Neighbors In-Sample Error: ", training_err)


		#Classify test points, get test error
		test_predictions = model.predict(test[:,1:])
		test_err = float(len(
			[
				test_predictions[i] 
				for i in range(len(test_predictions))
				if not test_predictions[i] == test[i,0]
			]
		)) / float(len(test_predictions))
		print("Test Error: ", test_err)

		#Plot crossval, test errors against lambda
		plt.plot(range(1,21), cross_val_errors, color='b')
		plt.show()
		plt.close()

		__plot_decision_boundary(D, test_predictions)


	#============================= RBF Network =================================

	if '--rbf' in sys.argv:

		#Partition D into training and test sets
		training = D[:300]
		test = D[300:]

		#iterate over k's and get cross-val errors
		best_k = None
		best_err = float('inf')
		cross_val_errors = [] # record errors for lambdas
		for test_k in range(1, 21):
			print("Testing k=", test_k)

			#Get total crossval error for lambda
			cv_err = __get_rbf_cv_err(test_k, training[:,1:], training[:,0])
			cross_val_errors.append(cv_err)

			#Update optimal crossval error, lamb
			if cv_err < best_err:
				best_err = cv_err
				best_k = test_k

		print("RBF Network Cross-validation error: ", best_err)
		print("Optimal K: ", best_k)

		#Use optimal k, get centers, weights and in-sample error
		kmeans = KMeans(n_clusters=best_k)
		kmeans.fit(X)
		radius = 2.0 / math.sqrt(float(best_k))
		Z_training = __build_gaussian_kernel_matrix(
						training[:,1:], 
						kmeans.cluster_centers_, 
						radius)
		weights, error_in = __run_rbf_pocket_algorithm(
							kmeans.cluster_centers_, 
							Z_training, 
							training[:,0], 
							300)

		print("RBF Network in-sample error: ", error_in)

		#Classify test points, get test error
		Z_test = __build_gaussian_kernel_matrix(
						test[:,1:], 
						kmeans.cluster_centers_, 
						radius)
		test_predictions = []
		for point in Z_test:
			test_predictions.append(np.dot(weights, point))
		test_predictions = np.sign(np.array(test_predictions))
		test_err = float(len(
			[
				test_predictions[i] 
				for i in range(len(test_predictions))
				if not test_predictions[i] == test[i][0]
			]
		)) / float(len(test_predictions))
		print("RBF Network Test Error: ", test_err)

		#Plot crossval, test errors against lambda
		plt.plot(np.arange(1, 21), cross_val_errors, color='b')
		plt.show()
		plt.close()

		__plot_decision_boundary(D, test_predictions)

	return 0

if __name__ == '__main__':
	sys.exit(main())