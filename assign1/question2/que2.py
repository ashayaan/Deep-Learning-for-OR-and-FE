# -*- coding: utf-8 -*-
# @Author: ashayaan
# @Date:   2019-09-21 00:37:27
# @Last Modified by:   ashayaan
# @Last Modified time: 2019-10-02 02:02:42
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from que2_network import Network
from sklearn.utils import shuffle
import itertools



epochs = 1000
learning_rate = 0.01

class Train(object):
	"""docstring for Train"""
	def __init__(self,epochs,learning_rate):
		super(Train, self).__init__()
		self.learning_rate = learning_rate
		self.net = Network()
		self.W = torch.ones(2,requires_grad=True)
		self.b = torch.ones(1,requires_grad=True)
		self.optimizer = torch.optim.Adagrad(itertools.chain(self.net.parameters(), [self.W], [self.b]), lr = self.learning_rate)
		self.loss_function =  torch.nn.BCELoss(reduction='sum')
		

def generatingSpace(t):
	r1 = 50 + 0.2 * t
	r2 = 30 + 0.4 * t
	phi1 = -0.06*t + 3
	phi2 = -0.08*t + 2

	#curve1
	x1 = r1 * np.cos(phi1)
	y1 = r1 * np.sin(phi1)

	#curve2
	x2 = r2 * np.cos(phi2)
	y2 = r2 * np.sin(phi2)
	# plt.plot(x1,y1)
	# plt.plot(x2,y2)
	# plt.show()

	return np.array([x1,y1,np.ones(x1.shape)]), np.array([x2,y2,np.zeros(x2.shape)])



def trainModel(model,df):
	total_loss = 0
	for i in range(len(df)):
		data = torch.tensor((df.iloc[i]['X'],df.iloc[i]['Y'])).float()
		label = torch.tensor(df.iloc[i]['label'])
		out = model.net.forward(data)
		# print (out)
		out = torch.sigmoid(torch.matmul(model.W,out) + model.b )
		loss = model.loss_function(out,label)
		model.optimizer.zero_grad()
		loss.backward()
		model.optimizer.step()

		total_loss += loss.item()

	return model,total_loss/len(df)


if __name__ == '__main__':
	t = np.arange(1,100)
	curve1,curve2 = generatingSpace(t)
	data = np.concatenate((curve1,curve2),axis=1)
	dataset = pd.DataFrame({'X': data[0,:], 'Y': data[1,:], 'label':data[2,:]})
	Transforming = dataset.copy()
	df = shuffle(dataset)	
	print (len(df))

	model = Train(epochs,learning_rate)
	
	for epoch in range(epochs):	
		df = shuffle(dataset)	
		model,total_loss = trainModel(model,df)
		print ("Epoch: {} Loss: {}".format(epoch+1,total_loss))

		
	#Transforming the curve
	x_curve1 = []
	y_curve1 = []
	
	x_curve2 = []
	y_curve2 = []

	curve3 = np.array([np.arange(0,1000000,100000),np.arange(0,1000000,10000)])
	x_curve3 = []
	y_curve3 = []	

	for i in (range(len(curve1[0]))):
		data = torch.tensor([curve1[0][i],curve1[1][i]]).float()
		out = model.net.forward(data)
		x_curve1.append(out[0].item())
		y_curve1.append(out[1].item())
		
		data = torch.tensor([curve2[0][i],curve2[1][i]]).float()
		out = model.net.forward(data)
		x_curve2.append(out[0].item())
		y_curve2.append(out[1].item())

		

	for i in (range(len(curve3[0]))):
		data = torch.tensor([curve3[0][i],curve3[1][i]]).float()
		out = model.net.forward(data)
		x_curve3.append(out[0].item())
		y_curve3.append(out[1].item())

	

	a,b = model.W[0].item(),model.W[1].item()
	c = model.b[0].item()
	print (a,b,c)
	print (type(a))

	x_curve3 = np.array(x_curve3)

	plt.plot(x_curve1,y_curve1)
	plt.plot(x_curve2,y_curve2)
	plt.plot(x_curve3,-(a*x_curve3+c)/b)
	
	plt.show()
	