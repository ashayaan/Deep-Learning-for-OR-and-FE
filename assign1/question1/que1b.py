# -*- coding: utf-8 -*-
# @Author: ashayaan
# @Date:   2019-09-21 00:12:21
# @Last Modified by:   ashayaan
# @Last Modified time: 2019-09-27 22:28:26
import torch
import numpy as np 
import pandas as pd
from math import pi as PI
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings("ignore")


learning_rate = 0.001
epochs = 100

def generatingSpace():
	x = np.linspace(-1,1,200)
	y = np.linspace(-1,1,200)
	y1 = -0.6 * np.sin(PI/2 + 3*x) - 0.35
	y2 = -0.6 * np.sin(PI/2 + 3*x) + 0.25
	
	for w11 in range(-3,4,1):
		for w12 in range(-3,4,1):
			for w21 in range(-3,4,1):
				for w22 in range(-3,4,1):
					for b1 in range(-1,1,1):
						for b2 in range(-1,1,1):
							x_hat = np.tanh(w11*x + w21*y + b1)
							y_hat = np.tanh(w11*x + w21*y + b2)

							curve1 = -0.6 * np.sin(PI/2 + 3*x_hat) - 0.35
							curve2 = -0.6 * np.sin(PI/2 + 3*x_hat) + 0.25
							
							x_hat = np.concatenate((x_hat,x_hat),axis=0)
							data = np.concatenate((curve1,curve2),axis=0)	
							ones = np.ones(data.shape)

							label1 = np.repeat(np.array([1]),curve1.shape[0])
							label2 = np.repeat(np.array([-1]),curve2.shape[0])
							labels = np.concatenate((label1,label2))
							
							df = pd.DataFrame({'X':x_hat[:],'Y':data[:], '1':ones[:],'labels':labels[:]})
							df = shuffle(df)
		
							W = torch.ones(3,requires_grad=True)
							optimizer = torch.optim.Adam([W],lr = learning_rate)


							print('\n')
							print('####################################')
							print('Running for ' + str(w11) + " " + str(w12) + " " + str(w21) + " " + str(w22) + " " + str(b1) + " " + str(b2))
							print('####################################')

							for epoch in range(epochs):
								l = 0
								for i in range(len(df)):
									label = torch.tensor([df.iloc[i]['labels']])
									eta = torch.tensor([df.iloc[i]['X'],df.iloc[i]['Y'],df.iloc[i]['1']])
									output = torch.matmul(W,eta)
									if output * label <= 0:
										l =  l + (-1 * output * label)
								optimizer.zero_grad()
								l.backward()
								optimizer.step()
								print ("Epoch: {} Loss: {}".format(epoch+1,l))


	# return x_hat,y_hat

if __name__ == '__main__':
	generatingSpace()
