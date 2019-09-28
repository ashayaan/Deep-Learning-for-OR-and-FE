# -*- coding: utf-8 -*-
# @Author: ashayaan
# @Date:   2019-09-21 15:41:09
# @Last Modified by:   ashayaan
# @Last Modified time: 2019-09-27 23:36:02
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

		self.fc1 = nn.Linear(2,12)
		self.fc2 = nn.Linear(12,18)
		self.fc3 = nn.Linear(18,8)
		self.fc4 = nn.Linear(8,4)
		self.fc5 = nn.Linear(4,2)

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