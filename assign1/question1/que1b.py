# -*- coding: utf-8 -*-
# @Author: ashayaan
# @Date:   2019-09-21 00:12:21
# @Last Modified by:   ashayaan
# @Last Modified time: 2019-09-24 14:57:02
import torch
import numpy as np 
import pandas as pd
from math import pi as PI
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings("ignore")

def generatingSpace():
	x = np.linspace(-1,1,200)
	y1 = -0.6 * np.sin(PI/2 + 3*x) - 0.35
	y2 = -0.6 * np.sin(PI/2 + 3*x) + 0.25
	
	for w11 in range(-3,4,1):
		for w12 in range(-3,4,1):
			for w21 in range(-3,4,1):
				for w22 in range(-3,4,1):
					for b1 in range(-1,1,1):
						for b2 in range(-1,1,1):
							lr = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
							curve1 = np.tanh(w11*x + w21*y1 + b1)
							curve2 = np.tanh(w12*x + w22*y2 + b2)

							data = np.concatenate((curve1,curve2))
							label1 = np.ones(y1.shape,dtype=np.float64)
							lable2 = np.zeros(y2.shape,dtype=np.float64)

							label = np.concatenate((label1,lable2),axis=0)
							data = np.concatenate((y1,y2),axis=0)

							dataset = pd.DataFrame({'X': data[:], 'label':label[:]})
							df = shuffle(dataset)	
							
							X = np.array(df['X']).reshape(-1,1)
							Y = np.array(df['label']).reshape(-1,1)
							lr.fit(data.reshape(-1,1),label.reshape(-1,1))
							print (lr.score(data.reshape(-1,1),label.reshape(-1,1)))

							plt.plot(x,curve1)
							plt.plot(x,curve2)
							w = lr.coef_.reshape(1)
							b = lr.intercept_.reshape(1)
							print (w,b)
							# print (w*x +b)
							plt.plot(x,w*x +b)
							plt.show()



	# return x_hat,y_hat

if __name__ == '__main__':
	generatingSpace()
