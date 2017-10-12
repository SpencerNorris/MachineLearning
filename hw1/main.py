#!/usr/bin/env python3

'''
Author: Spencer Norris
File: main.py
Description: Implementation of a single-layer perceptron that ingests
tabular data, or can generate a random dataset. This dataset is then
used to attempt to find a decision boundary, which is then plotted.
'''
import matplotlib.colors as colors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from random import shuffle
from copy import copy
import pandas as pd
import numpy as np
import random
import sys



def TARGET_FUNCTION(x):
	TRUE_WEIGHTS=np.array([-1, .5])
	return np.dot(x, TRUE_WEIGHTS) + 4

def SIGN(x):
	return 1 if TARGET_FUNCTION(x) > 0 else -1

class Perceptron():
	def __init__(self, dims=2):
		self.weights = np.random.uniform(.01, .05, dims + 1)

	def __update_weights(self, example, true_label):
		self.weights = self.weights + int(true_label)*np.append(example, [1])

	def train(self, data):
		num_examples = len(data.index)
		updates = 0

		#Wait until convergence
		while True:

			#Iterate over all examples
			#If we misclassify, restart
			#Otherwise, count correct classifications
			shuffle(data.values)
			correct_classifications = 0
			for row in data.values:
				true_label = row[-1]
				assert true_label == SIGN(row[:-1])
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



def main(data):
	x_min = min(data['x1'])
	x_max = max(data['x1'])

	#Train perceptron
	perceptron = Perceptron()
	updates = perceptron.train(copy(data))
	print("NUMBER OF UPDATES: ", updates)

	#Generate base figure
	fig, ax = plt.subplots()
	colors = { -1: 'red', 1 : 'blue'}
	positives = data[data.y==1]
	negatives = data[data.y==-1]
	plot_positives = ax.scatter(positives['x1'], positives['x2'], color='red',s=5)
	plot_negatives = ax.scatter(negatives['x1'], negatives['x2'], color='blue',s=5)
	line_func = lambda x : 2*x - 8
	line = mlines.Line2D([x_min, x_max], [line_func(x_min), line_func(x_max)], color='red')
	ax.add_line(line)

	#Add preceptron to diagram
	weights = perceptron.weights
	w_0 = weights[-1]
	w_1 = weights[0]
	w_2 = weights[1]
	print("w_0", w_0)
	print("w_1", w_1)
	print("w_2", w_2)
	perceptron_boundary = lambda x1 : -1*(w_1/w_2)*x1 - (w_0/w_2)
	perceptron_line =  mlines.Line2D(
					[x_min, x_max], 
					[perceptron_boundary(x_min), perceptron_boundary(x_max)], 
					color='green')
	perceptron_str = 'g(x) '
	ax.add_line(perceptron_line)

	#Set legend, labels
	ax.set_xlabel('x1')
	ax.set_ylabel('x2')
	ax.legend(
			(plot_positives, plot_negatives, line, perceptron_line),
			('+1', '-1', 'x2 = 2(x1) - 8', perceptron_str)
		)
	plt.show()
	return 0


if __name__ == '__main__':
	if not '--random' in sys.argv:
		sys.exit(main(data=pd.read_csv('./linearly_separable_data.csv')))
	else:
		num_examples = int(sys.argv[sys.argv.index('--random') + 1])
		data = pd.DataFrame(
			np.random.uniform(
				low=-100,
				high=100,
				size=(num_examples,3)
			),
			columns='x1,x2,y'.split(','))
		data['y'] = data.apply(lambda row: SIGN(row[:-1]), axis=1)
		print(data)
		sys.exit(main(data=data))
