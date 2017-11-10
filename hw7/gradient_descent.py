#!/usr/bin/env python3
'''
Author: Spencer Norris
File: gradient_descent.py
Description: Illustration of gradient descent in action on a sample function.
'''

import matplotlib.pyplot as plt
import numpy as np
import math
import sys

x_0 = None
y_0 = None
eta = None

def fn(x,y):
	return x**2 + y**2 + 2 * math.sin(math.degrees(x * math.pi)) * math.sin(math.degrees(y * math.pi))

def gradient_fn(x,y):
	ddx_fn = lambda x,y: 2 * (math.pi * math.cos(math.degrees(x*math.pi)) * math.sin(math.degrees(y*math.pi)) + x)
	ddy_fn = lambda x,y: 2 * (math.pi * math.sin(math.degrees(x*math.pi)) * math.cos(math.degrees(y*math.pi)) + y)
	return [ddx_fn(x,y), ddy_fn(x,y)]

def main():
	global x_0
	global y_0
	x, y = x_0, y_0
	update_values = []
	update_values.append(fn(x,y))

	#Perform gradient descent
	for i in range(50):
		update = gradient_fn(x,y)
		mag = math.sqrt(update[0]**2 + update[1]**2)
		x = x - eta * float(update[0] / mag)	
		y = y - eta * float(update[1] / mag)
		update_values.append(fn(x,y))

	print("Final X: ", x)
	print("Final Y: ", y)
	print("Final value: ", fn(x,y))

	#Plot fn over course of updates
	plt.scatter(np.arange(0,51), update_values, color='blue')
	plt.show()

if __name__ == '__main__':
	x_0 = float(sys.argv[1])
	y_0 = float(sys.argv[2])
	eta = float(sys.argv[3])
	sys.exit(main())