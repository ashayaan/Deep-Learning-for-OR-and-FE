# -*- coding: utf-8 -*-
# @Author: ashayaan
# @Date:   2019-09-21 00:37:27
# @Last Modified by:   ashayaan
# @Last Modified time: 2019-09-24 11:44:35
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from que2_network import Network
from sklearn.utils import shuffle


epochs = 1000

class Train(object):
	"""docstring for Train"""
	def __init__(self,epochs,learning_rate):
		super(Train, self).__init__()
		self.learning_rate = learning_rate
		self.net = Network()
		self.parameters = self.net.parameters()
		self.optimizer = torch.optim.SGD(self.parameters,lr = self.learning_rate)
		self.loss_function =  torch.nn.BCELoss(reduction='mean')
		
		

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

	plt.plot(x1,y1)
	plt.plot(x2,y2)
	plt.show()

	return np.array([x1,y1,np.ones(x1.shape)]), np.array([x2,y2,np.zeros(x2.shape)])



def trainModel(model,df):

	total_loss = 0
	for i in range(df.shape[0]):
		data = torch.tensor((df.iloc[i]['X'],df.iloc[i]['Y'])).float()
		label =  torch.tensor([df.iloc[i]['label']]).float()
		out = model.net.forward(data)
		model.optimizer.zero_grad()
		loss = model.loss_function(out,label)
		loss.backward()
		model.optimizer.step()
		
		total_loss += loss.item()

	total_loss /= df.shape[0]
	return model,total_loss

if __name__ == '__main__':
	t = np.arange(1,1001)
	curve1,curve2 = generatingSpace(t)
	print (curve1.shape,curve2.shape)
	data = np.concatenate((curve1,curve2),axis=1)
	print (data.shape)
	model = Train(100,0.01)
	dataset = pd.DataFrame({'X': data[0,:], 'Y': data[1,:], 'label':data[2,:]})
	df = shuffle(dataset)	
	# print (df)
	for epoch in range(epochs):
		print('Epoch: {}'.format(epoch+1))
		df = shuffle(dataset)	
		model,total_loss = trainModel(model,df)
		print ("Epoch: {} Loss: {}".format(epoch+1,total_loss))