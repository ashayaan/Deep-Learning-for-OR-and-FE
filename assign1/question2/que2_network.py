# -*- coding: utf-8 -*-
# @Author: ashayaan
# @Date:   2019-09-21 15:41:09
# @Last Modified by:   ashayaan
# @Last Modified time: 2019-09-24 11:47:41
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

class Network(nn.Module):
	"""docstring for Network"""
	def __init__(self):
		super(Network, self).__init__()
		self.relu = nn.ReLU()
		self.Sigmoid = nn.Sigmoid()

		self.fc1 = nn.Linear(2,20)
		self.fc2 = nn.Linear(20,15)
		self.fc3 = nn.Linear(15,10)
		self.fc4 = nn.Linear(10,5)
		self.fc5 = nn.Linear(5,1)

	def forward(self,data):
		out = self.relu(self.fc1(data))
		out = self.relu(self.fc2(out))
		out = self.relu(self.fc3(out))
		out = self.relu(self.fc4(out))
		out = self.Sigmoid(self.fc5(out))
		return out



if __name__ == '__main__':
	#Unit Testing
	data = torch.rand(2)
	test = Network()
	print (test.forward(data))