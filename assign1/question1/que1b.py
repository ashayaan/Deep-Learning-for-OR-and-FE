# -*- coding: utf-8 -*-
# @Author: ashayaan
# @Date:   2019-09-28 16:34:40
# @Last Modified by:   ashayaan
# @Last Modified time: 2019-09-28 16:43:46


import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1,1,200)
y = np.linspace(-1,1,200)

y1 = -0.6*np.sin(np.pi/2 + 3*x) - 0.35
y2 = -0.6*np.sin(np.pi/2 + 3*x) + 0.25

def generatingSpace():
	for w11 in range(-3,3):
		for w12 in range(-3,3):
			for w21 in range(-3,3):
				for w22 in range(-3,3):
					for b1 in range(-1,2):
						for b2 in range(-1,2):
							for a in range(-6,6):
								for b in range(-6,6):
									for c in range(-6,6):
										w = np.array([a,b,c])
										Flag = True
										
										x_hat = np.tanh(w11*x + w21*y + b1)
										y_hat = np.tanh(w12*x + w22*y + b2)

										x_hat_curve_1 = np.tanh(w11*x + w21*y1 + b1)
										y_hat_curve_1 = np.tanh(w12*x + w22*y1 + b2)

										x_hat_curve_2 = np.tanh(w11*x + w21*y2 + b1)
										y_hat_curve_2 = np.tanh(w12*x + w22*y2 + b2)

										for (i,j) in zip(x_hat_curve_1, y_hat_curve_1):
											product = np.dot(w, np.array([i,j,1]))
											if product >= 0:
												Flag = False
												break

										if Flag==True:
											for (i,j) in zip(x_hat_curve_2, y_hat_curve_2):

												product = np.dot(w, np.array([i,j,1]))
												if product < 0:
													Flag = False
													break
													
										if Flag==True:
											return w11, w12, w21, w22, b1, b2, a, b, c
											
											
											

if __name__ == '__main__':
	w11,w12, w21, w22, b1, b2, a, b, c = generatingSpace()

	print (w11,w12, w21, w22, b1, b2, a, b,c)

	x_hat_curve_1 = np.tanh(w11*x + w21*y1 + b1)
	y_hat_curve_1 = np.tanh(w12*x + w22*y1 + b2)

	x_hat_curve_2 = np.tanh(w11*x + w21*y2 + b1)
	y_hat_curve_2 = np.tanh(w12*x + w22*y2 + b2)

	x_hat = np.tanh(w11*x + w21*y + b1)

	plt.plot(x_hat_curve_1,y_hat_curve_1)
	plt.plot(x_hat_curve_2,y_hat_curve_2)
	plt.plot(x_hat,-(a*x_hat+c)/b)
	plt.show()