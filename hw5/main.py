#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import random
import sys


def __get_random_function():
	x_1 = random.uniform(-1, 1)
	x_2 = random.uniform(-1, 1)
	y_1 = x_1**2
	y_2 = x_2**2
	a = None
	try:
		a = ( (y_2 - y_1) / (x_2 - x_1) )
	except ZeroDivisionError:
		a = 0
	b = y_1 - ((y_2 - y_1)/(x_2 - x_1)) * x_1
	return a, b, lambda x: a*x + b


def main():
	#Compute average a, b
	avg_a = 0.0
	avg_b = 0.0
	for i in range(10000):
		a, b, func = __get_random_function()
		avg_a += a
		avg_b += b
	avg_a = avg_a / 10000.0
	avg_b = avg_b / 10000.0
	print("avg_a: ", avg_a)
	print("avg_b: ", avg_b)


	f = lambda x: x**2
	g_avg = lambda x: avg_a*x + avg_b

	#Estimate bias
	bias = 0.0
	for i in range(10000):
		rand_x = random.uniform(-1,1)
		bias += ( g_avg(rand_x) - f(rand_x) )**2
	bias = bias / 10000.0
	print("Est. Bias: ", bias)

	#Estimate variance
	variance = 0.0
	for i in range(10000):
		rand_x = random.uniform(-1,1)
		a,b,g = __get_random_function()
		variance += (g(rand_x) - g_avg(rand_x))**2
	variance = variance / 10000.0
	print("Est. Variance: ", variance)

	#Empirically determine E_out
	avg_e_out = 0.0
	
	#Randomly select 10000 different random functions
	for i in range(10000):
		a,b,g = __get_random_function()
		#Evaluate error between f(x) and each random function on 10000 random points
		e_out = 0.0
		for j in range(10000):
			x_rand = random.uniform(-1,1)
			e_out += (g(x_rand) - f(x_rand))**2
		e_out = e_out / 10000.0
		avg_e_out += e_out
	avg_e_out = avg_e_out / 10000.0
	print("avg. E_out: ", avg_e_out)

	print("Sum of estimated bias and variance: ", bias + variance)

	#Plot average function
	X = np.arange(-1,1,.001)
	Y_f = [f(x) for x in X]
	Y_g = [g_avg(x) for x in X]
	plt.plot(X, Y_f, 'b')
	plt.plot(X, Y_g, 'r')
	plt.show()

	return 0


if __name__ == '__main__':
	sys.exit(main())