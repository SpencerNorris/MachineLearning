#!/usr/bin/env python3
'''
Author: Spencer Norris
File: main.py
Desc: Seriously, it's machine learning homework. What do you want?
'''

import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import math
import sys


def coins(num_runs):
	'''
	Flips 1,000 fair coins 10 times each.
	Runs this experiment num_runs times.
	Plots a histogram of the values assumed by
	first coin flipped (c_1), a randomly selected coin
	(c_rand), and the coin with the least number of heads (c_min).
	'''

	def __flip_coin():
		val = random.uniform(0,1)
		return 1 if val <= .5 else 0

	#Begin experiment
	v_1_vals = []
	v_rand_vals = []
	v_min_vals = []
	for run in range(num_runs):

		#Flip all 1,000 coins 10 times each
		coins = [0 for x in range(1000)]
		for coin in range(1000):
			for flip in range(10):
				coins[coin] += __flip_coin()

		#Select v_1, v_rand and v_min, add vals to collections
		v_1 = float(coins[0]) / 10.0
		v_rand = float(random.choice(coins)) / 10.0
		v_min = float(min(coins)) / 10.0

		v_1_vals.append(v_1)
		v_rand_vals.append(v_rand)
		v_min_vals.append(v_min)


	print("avg. v1: ", float(sum(v_1_vals)) / float(len(v_1_vals)) )
	print("avg. v_rand: ", float(sum(v_rand_vals)) / float(len(v_rand_vals)) )
	print("avg. v_min: ", float(sum(v_min_vals)) / float(len(v_min_vals)) )


	datasets = [v_1_vals, v_rand_vals, v_min_vals]

	#Create histograms
	fig = plt.figure()
	gs = gridspec.GridSpec(3, 2)
	ax_1 = fig.add_subplot(gs[0, 0])
	ax_rand = fig.add_subplot(gs[1, 0])
	ax_min = fig.add_subplot(gs[2, 0])
	ax_1.set_title("Coin 1")
	ax_rand.set_title("Random Coin")
	ax_min.set_title("Minimum Coin (Fewest Heads)")


	#Create Hoeffding curves, P(|v - mu| > eps) points
	ax_1_error = fig.add_subplot(gs[0,1])
	ax_rand_error = fig.add_subplot(gs[1,1])
	ax_min_error = fig.add_subplot(gs[2,1])

	mu = .5
	for dataset, ax in zip(datasets, (ax_1_error, ax_rand_error, ax_min_error)):
		epsilon = np.linspace(0.0, 1.0, num=1000)
		estimate = [(lambda x : float(len([v for v in dataset 
						if abs(v - mu)  >= x])))(eps) / float(len(dataset))
						for eps in epsilon]
		ax.plot(epsilon, estimate)
		print(sorted(estimate))
		hoeffding = [(lambda x: 2*math.exp(-2 * len(dataset) * math.pow(x,2)))(eps)
						for eps in epsilon]		
		ax.plot(epsilon, hoeffding)

	BOUNDS = np.linspace(0.0, 1.0, num=20)
	ax_1.hist(v_1_vals, BOUNDS, histtype='bar', align='right', )
	ax_rand.hist(v_rand_vals, BOUNDS, histtype='bar', align='right')
	ax_min.hist(v_min_vals, BOUNDS, histtype='bar', align='right')
	gs.tight_layout(fig)
	plt.show()	plt.show()
	return 0


if __name__ == '__main__':
	runs = int(sys.argv[1])
	coins(runs)