#!/usr/bin/env python3
'''
Author: Spencer Norris
Just plots the bounds for the generalization error
when using E_in and E_test, respectively.
'''


import matplotlib.pyplot as plt
import numpy as np
import math
import sys


def __etest(N):
	return math.sqrt( (1/(2*N)) * math.log(2 / 0.05) )

def __ein(N):
	return math.sqrt( (8/N) * math.log(8 / 0.05) )

def main():
	N = np.arange(1.0, 601.0, 1.0)
	N_rev = list(reversed(N))

	plt.plot(N, [__ein(x) for x in N], 'b')
	plt.plot(N, [__etest(x) for x in N_rev], 'r')
	plt.show()

if __name__ == '__main__':
	sys.exit(main())
