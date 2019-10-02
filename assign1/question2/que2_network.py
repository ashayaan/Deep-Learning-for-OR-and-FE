# -*- coding: utf-8 -*-
# @Author: ashayaan
# @Date:   2019-09-21 15:41:09
# @Last Modified by:   ashayaan
# @Last Modified time: 2019-10-02 01:02:36
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

class Network(nn.Module):
	"""docstring for Network"""
	def __init__(self):
		super(Network, self).__init__()
		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()
		self.Sigmoid = nn.Sigmoid()

		self.fc1 = nn.Linear(2,100,bias=True)
		self.fc2 = nn.Linear(100,50,bias=True)
		self.fc3 = nn.Linear(50,2)

	def forward(self,data):
		out = self.tanh(self.fc1(data))
		out = self.tanh(self.fc2(out))
		out = self.tanh(self.fc3(out))
		return out



if __name__ == '__main__':
	#Unit Testing
	data = torch.rand(2)
	test = Network()
	print (test.forward(data))