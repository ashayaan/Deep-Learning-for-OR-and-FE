# -*- coding: utf-8 -*-
# @Author: ashayaan
# @Date:   2019-09-21 00:09:14
# @Last Modified by:   ashayaan
# @Last Modified time: 2019-09-27 14:11:52

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from math import pi as PI
import matplotlib.pyplot as plt
from que1_params import learning_rate
from que1_params import epochs
import pandas as pd

from sklearn.utils import shuffle

class LinearSperator():
	"""docstring for LinearSperator"""
	def __init__(self,learning_rate):
		self.learning_rate = learning_rate
		self.W = torch.ones(1,dtype=torch.float64,requires_grad=True)
		self.B = torch.ones(1,dtype=torch.float64,requires_grad=True)
		self.optimizer = torch.optim.SGD([self.W,self.B],lr = self.learning_rate)
		self.loss = nn.BCELoss(reduction='mean')


	def forward(self,data):
		return torch.sigmoid(self.W*data + self.B)
	

	def train(self,data,label):
		output = self.forward(data)
		loss = self.loss(output,label)
		print (loss.item())
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

if __name__ == '__main__':
	x = np.linspace(-1,1,1000)
	y1 = -0.6 * np.sin(PI/2 + 3*x) - 0.35
	y2 = -0.6 * np.sin(PI/2 + 3*x) + 0.25
	label1 = np.ones(y1.shape,dtype=np.float64)
	lable2 = np.zeros(y2.shape,dtype=np.float64)


	label = np.concatenate((label1,lable2),axis=0)
	data = np.concatenate((y1,y2),axis=0)

	
	df = pd.DataFrame({'data':data.tolist(),'label':label.tolist()})
	df = shuffle(df)
	data = np.array(df['data']).reshape(-1,1)
	label = np.array(df['label']).reshape(-1,1)

	print (data)

	data = torch.from_numpy(data)
	label = torch.from_numpy(label)
	test = LinearSperator(learning_rate)
	
	for epoch in range(epochs):
		test.train(data,label)


	y = test.W.item()*x + test.B.item()

	plt.plot(x,y1)
	plt.plot(x,y2)
	plt.plot(x,y)

	plt.show()