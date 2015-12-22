import numpy as np
import pandas as pd
#import matplotlib.pyplot  as plt
#from scipy import optimize
from make_data import make_data


def logistic(x, w, b):
	def sig(a):
		return 1.0 / (1.0 + np.exp(-a))

	return sig(np.dot(x, w) + b)

def f_gradient(x, y, w, b):
	error = y - logistic(x, w, b)
	w_grad = -np.mean(x.T * error, axis=1)
	b_grad = -np.mean(error)
	return w_grad, b_grad

def optimize(x, y, w, b, eta = 0.01, num = 1000):
	for i in range(1, num):
		w_grad, b_grad = f_gradient(x, y, w, b)
		w -= eta * w_grad
		b -= eta * b_grad
		e = np.mean(np.abs(y - p_y_given_x(x, w, b)))
		yield i, w, b, e

if __name__ == '__main__':
	df = make_data(100)
	x = np.array([df["x1"],df["x2"]]).T
	y = np.array(df["y"])
	w, b = np.zeros(2), 0

	optimize(x, y, w, b)
