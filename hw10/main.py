#!/usr/bin/env python3

'''
Author: Spencer Norris

Description: adaptation of code at http://bit.ly/2AWVSHO
to perform K-nearest neighbors and to plot decision boundaries.
Augmented to allow a --transform flag which maps each
point X into a feature space --> (sqrt(x_1**2 + x_2**2), arctan(x_2/x_1)).

Date: 11/20/17
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import math
import sys


transform = False

def main(path):
	global transform

	#Load dataset
	D = np.genfromtxt(path, delimiter=',')
	X = D[:, :len(D.T) - 1]
	Y = D[:, len(D.T) - 1]

	if transform:
		euc = lambda X: math.sqrt(X[0]**2 + X[1]**2)
		arctandiv = lambda X: np.arctan(X[1] / X[0])
		for i in range(len(X)):
			X[i] = np.array([euc(X[i]), arctandiv(X[i])])

	h = .02  # step size in the mesh

	# Create color maps
	cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
	cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

	for knn in [1, 3]:
	    #Create KNN classifier and fit
	    clf = neighbors.KNeighborsClassifier(knn, weights='uniform')
	    clf.fit(X, Y)

	    # Plot the decision boundary
	    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
	                         np.arange(y_min, y_max, h))
	    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

	    #Plot points
	    Z = Z.reshape(xx.shape)
	    plt.figure()
	    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
	    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap_bold,
	                edgecolor='k', s=20)
	    plt.xlim(xx.min(), xx.max())
	    plt.ylim(yy.min(), yy.max())
	    plt.title("%i-NN" % (knn))
	    plt.show()
	    plt.close()

if __name__ == '__main__':
	path = sys.argv[1]
	if '--transform' in sys.argv:
		transform = True
	sys.exit(main(path))

	