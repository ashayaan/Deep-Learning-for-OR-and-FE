# -*- coding: utf-8 -*-
# @Author: ashayaan
# @Date:   2019-09-21 00:12:21
# @Last Modified by:   ashayaan
# @Last Modified time: 2019-09-29 15:13:53
import torch
import numpy as np 
import pandas as pd
from math import pi as PI
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn import svm
import warnings
warnings.filterwarnings("ignore")



x = np.linspace(-1,1,200)
y = np.linspace(-1,1,200)
y1 = -0.6 * np.sin(PI/2 + 3*x) - 0.35
y2 = -0.6 * np.sin(PI/2 + 3*x) + 0.25

def generatingSpace():
	for w11 in range(-3,4,1):
		for w12 in range(-3,4,1):
			for w21 in range(-3,4,1):
				for w22 in range(2,4,1):
					for b1 in range(0,2,1):
						for b2 in range(0,2,1):
							clf = svm.SVC(kernel='linear',gamma='auto')
							lr = LogisticRegression()
							x_hat_curve_1 = np.tanh(w11*x + w21*y1 + b1)
							y_hat_curve_1 = np.tanh(w12*x + w22*y1 + b2)

							x_hat_curve_2 = np.tanh(w11*x + w21*y2 + b1)
							y_hat_curve_2 = np.tanh(w12*x + w22*y2 + b2)


							x_hat = np.concatenate((x_hat_curve_1,x_hat_curve_2),axis=0)
							y_hat = np.concatenate((y_hat_curve_1,y_hat_curve_2),axis=0)	
			

							label1 = np.repeat(np.array([1]),x_hat_curve_1.shape[0])
							label2 = np.repeat(np.array([-1]),x_hat_curve_2.shape[0])
							labels = np.concatenate((label1,label2))
							
							df = pd.DataFrame({'X':x_hat[:],'Y':y_hat[:],'labels':labels[:]})
							df = shuffle(df)
								

							data = np.array([df['X'],df['Y']]).reshape(400,2)
							labels = np.array(df['labels'])
							
							clf.fit(data,labels) 
							lr.fit(data,labels)

							print ("SVM SCORE " + str(clf.score(data,labels)))
							print ("LR SCORE " + str(lr.score(data,labels)))
							
							if (clf.score(data,labels) > 0.57):
								return w11,w12, w21, w22, b1, b2, lr.coef_[0],lr.intercept_[0]
							# print (clf.coef_)

if __name__ == '__main__':
	w11,w12, w21, w22, b1, b2, coef, intercept = generatingSpace()

	print (w11,w12, w21, w22, b1, b2, coef, intercept)

	a,b = coef
	c = intercept

	x_hat_curve_1 = np.tanh(w11*x + w21*y1 + b1)
	y_hat_curve_1 = np.tanh(w12*x + w22*y1 + b2)

	x_hat_curve_2 = np.tanh(w11*x + w21*y2 + b1)
	y_hat_curve_2 = np.tanh(w12*x + w22*y2 + b2)

	x_hat = np.tanh(w11*x + w21*y + b1)

	plt.plot(x_hat_curve_1,y_hat_curve_1)
	plt.plot(x_hat_curve_2,y_hat_curve_2)
	plt.plot(x_hat,-(a*x_hat+c)/b)
	plt.show()
