#!/usr/bin/env python3
'''
Calculates, given C coins and a probability mu of getting heads,
the likelihood of observing at least one coin with no heads.
'''

import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from scipy.special import comb
import numpy as np
import random
import math
import sys

import numpy
numpy.seterr(all='raise')

global N

def __prob_no_heads(mu):
	global N
	return math.pow(1 - mu, N)

def __get_prob_no_heads_all_coins(mu, C):
	vals = [ comb(C, c) * math.pow(__prob_no_heads(mu), c) for c in range(1,C+1) ]
	return sum([vals[i] * -1.0 if i % 2 == 1 else vals[i] for i in range(len(vals))])


def __flip_coin(mu):
	val = random.uniform(0,1)
	return 1 if val <= mu else 0

def plot_max(C, mu):
	#Plot estimate
	global N
	epsilon = np.linspace(0.0, 1.0, num=1000)
	d_est = []
	for d in range(1000):
		coin_heads = []
		for c in range(C):
			heads = 0
			for n in range(N):
				heads += __flip_coin(mu)
			coin_heads.append(heads)
		max_coin = max(coin_heads)
		d_est.append(float(max_coin) / float(N))
	estimate = [(lambda x : float(len([v for v in d_est 
						if abs(v - mu)  >= x])))(eps) / float(len(d_est))
					for eps in epsilon]
	plt.plot(epsilon, estimate)

	#Plot hoeffding
	hoeffding = [(lambda x: 2*math.exp(-2 * len(d_est) * math.pow(x,2)))(eps)
					for eps in epsilon]		
	plt.plot(epsilon, hoeffding)
	plt.show()


def main(C, mu):
	print(C)
	print(mu)
	return __get_prob_no_heads_all_coins(mu,C)


if __name__ == '__main__':
	global N
	coins = int(sys.argv[1])
	mu = float(sys.argv[2])
	N = int(sys.argv[3])
	if not '--plot-max' in sys.argv:
		print("RESULT: ", main(coins, mu))
	else:
		plot_max(coins,mu)