#!/usr/bin/env python3

'''
Author: Spencer Norris

Description: adaptation of code at http://bit.ly/2AWVSHO
Essentially the same as main.py, only performed on
autogenerated semicircles and with no feature transform.

Date: 11/20/17
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import random
import math
import sys

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

	radius = 10
	thickness = 5
	separation = 5

	#Generate datasets
	top_data = np.array([__top_semicircle(radius,thickness) for i in range(1000)])
	top_data = np.append(top_data, [[1] for i in range(len(top_data))], axis=1)

	bottom_data = np.array([__bottom_semicircle(radius,thickness,separation) for i in range(1000)])
	bottom_data = np.append(bottom_data, [[-1] for i in range(len(top_data))], axis=1)

	D = np.concatenate((top_data, bottom_data))
	X = D[:, :len(D.T) - 1]
	Y = D[:, len(D.T) - 1]

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
	sys.exit(main())