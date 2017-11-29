#!/usr/bin/env python3

'''
Author: Spencer Norris

Description: implementation of branch-and-bound
and brute force K nearest neighbor approaches
in order to compare performance on 10,000
randomly generated points.

Date: 11/20/2017
'''

from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from copy import copy
import numpy as np
import random
import timeit
import sys


brute = False
gaussian = False

def __partition_data(D):
	'''
	Partitions data into 10 different groups based on closeness to
	centers selected to maximize distance from each other.
	'''
	centers = []
	dist_matrix = np.zeros((len(D), len(D)))

	#Calculate distances
	for i in range(len(D)):
		for j in range(len(D)):
			dist_matrix[i][j] = np.linalg.norm(D[i][:len(D.T) - 1] - D[j][:len(D.T) - 1])


	#Select centers (10 total)
	for k in range(10):

		#Randomly select first center
		if k == 0:
			first = random.randrange(len(D))
			centers.append(first)

		#Find furthest point based on minimum distance of that point from another center
		else:
			index = -1
			best_dist = float('-inf')
			for i in range(len(dist_matrix)):
				if i in centers: #Check if we already added the row to centers
					continue
				else:
					min_dist = min([dist_matrix[i][j] for j in centers])
					if min_dist > best_dist:
						index = i
						best_dist = min_dist
			centers.append(index)


	#Partition data according to distance from each center
	partitions = [None for k in range(len(centers))]
	for i in range(len(D)):
		dists = [(center, np.linalg.norm(D[center][:len(D.T) - 1] - D[i][:len(D.T) - 1])) for center in centers]
		closest = min(dists, key=lambda x: x[1])[0]
		index = centers.index(closest)
		if partitions[index] is None:
			partitions[index] = [i]
		else:
			partitions[index].append(i)

	return partitions, dist_matrix


def __brute_classify(D):
	'''
	Performs a brute-force search across all nodes
	and their distances from each other.
	'''
	#Calculate distances
	dist_matrix = np.zeros((len(D), len(D)))
	for i in range(len(D)):
		for j in range(len(D)):
			dist_matrix[i][j] = np.linalg.norm(D[i][:len(D.T) - 1] - D[j][:len(D.T) - 1])

	#Classify nodes according to closest neighbor's distance (yes, I know I'm re-using the training set, 
	#but this is only to demonstrate the approx. running time, so sue me.)
	classifications = []
	for i in range(len(D)):
		closest_index = -1
		closest_dist = float('inf')
		for j in range(len(D[i])):
			if dist_matrix[i][j] < closest_dist:
				closest_index = j
				closest_dist = dist_matrix[i][j]
		classifications.append(D[closest_index, len(D.T) - 1])
	return classifications


def __branch_and_bound_classify(D, partitions, centers, radii, dist_matrix):
	'''
	Perform k-nearest neighbors classification using the branch-and-bound optimization.
	'''
	classifications = []
	for example in range(len(D)):

		#Get ordered dist of clusters, from lowest to highest
		cluster_order = sorted(
				[index for index in range(len(centers))],
				key = lambda index: np.linalg.norm(D[example][:2] - centers[index])
			)

		#Search clusters until bound condition is satisfied
		nearest_dist = float('inf')
		nearest_index = -1
		for index in cluster_order:

			#Check if we need to keep searching
			if nearest_dist <= np.linalg.norm(D[example][:2] - centers[index]) - radii[index]:
				break
			else:
				#Find closest within partition
				for elem in partitions[index]:
					if dist_matrix[example][elem] < nearest_dist:
						nearest_dist = dist_matrix[example][elem]
						nearest_index = elem
		classifications.append(D[elem][len(nearest) - 1])
	return classifications


def main():

	global brute
	global gaussian
	
	#Generate data and partition
	if gaussian:
		mix = GaussianMixture(
				n_components=10,
				covariance_type='diag',
				means_init=[random.random() for i in range(10)]
			)
		D = np.array([
			mix.predict([random.random(), random.random()])
			for i in range(10000)])
		print(D)
		sys.exit(0)
	else:
		D = np.array([
				[
					random.random(), 
					random.random(), 
					-1 if random.random() < .5 else 1
				]
			for i in range(10000)])


	#Run appropriate version of k nearest neighbors
	start = timeit.default_timer()
	if brute:
		__brute_classify(D)
		
	else:
		#Partition data
		partitions, dist_matrix = __partition_data(D)

		#Compute centers of each partition
		centers = np.array([
				np.mean(np.array([D[i] for i in partitions[p]]), axis=0)[:2] 
			for p in range(len(partitions))])

		#Compute radii of each cluster
		radii = [
				max([
					np.linalg.norm(centers[p] - partitions[p][i][:2]) 
					for i in range(len(partitions[p]))
				]) 
		for p in range(len(centers))]

		#Perform k-means classification with branch and bound
		__branch_and_bound_classify(D, partitions, centers, radii, dist_matrix)

	stop = timeit.default_timer()

	print("Total runtime: ", stop - start)
	return 0


if __name__ == '__main__':
	if '--brute' in sys.argv:
		brute = True
	if '--gaussian' in sys.argv:
		gaussian = True
	sys.exit(main())