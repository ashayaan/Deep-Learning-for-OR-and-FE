import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from math import pi as PI
import matplotlib.pyplot as plt
from que1_params import learning_rate
from que1_params import epochs

from sklearn.linear_model import LogisticRegression


class LinearSperator():
	"""docstring for LinearSperator"""
	def __init__(self,learning_rate):
		self.learning_rate = learning_rate
		self.W = torch.zeros(1,dtype=torch.float64,requires_grad=True)
		self.B = torch.zeros(1,dtype=torch.float64,requires_grad=True)
		self.optimizer = torch.optim.SGD([self.W,self.B],lr = self.learning_rate)
		self.loss =  torch.nn.BCELoss(reduction='mean')


	def forward(self,data):
		return F.sigmoid(self.W*data + self.B)
	

	def train(self,data,label):
		output = self.forward(data)
		loss = self.loss(output,label)
		print (loss.item())
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

if __name__ == '__main__':
	x = np.linspace(-10* np.pi,10 * np.pi,1000)
	y1 = -0.6 * np.sin(PI/2 + 3*x) - 0.35
	y2 = -0.6 * np.sin(PI/2 + 3*x) + 0.25
	label1 = np.zeros(y1.shape,dtype=np.float64)
	lable2 = np.ones(y2.shape,dtype=np.float64)

	label = np.concatenate((label1,lable2),axis=0).reshape(2000,1)
	data = np.concatenate((y1,y2),axis=0).reshape(2000,1)	

	
	data = torch.from_numpy(data)
	label = torch.from_numpy(label)
	test = LinearSperator(learning_rate)
	
	for epoch in range(epochs):
		test.train(data,label)
		# print ('Epoch ' + str(epoch) + ' Loss ' + str(loss.item()))

	# print () 
	y = test.W.item()*x + test.B.item()

	# print (y.shape)

	plt.plot(x,y1)
	plt.plot(x,y2)
	plt.plot(x,y)

	plt.show()