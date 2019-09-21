import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from math import pi as PI
import matplotlib.pyplot as plt
from que1_params import learning_rate
from que1_params import epochs

class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        out = self.linear(x)
        return out
	


if __name__ == '__main__':
	x = np.linspace(-10* np.pi,10 * np.pi,1000)
	y1 = -0.6 * np.sin(PI/2 + 3*x) - 0.35
	y2 = -0.6 * np.sin(PI/2 + 3*x) + 0.25
	label1 = np.zeros(y1.shape,dtype=np.float64)
	lable2 = np.ones(y2.shape,dtype=np.float64)

	label = np.concatenate((label1,lable2),axis=0)
	data = np.concatenate((y1,y2),axis=0)	


	data = torch.from_numpy(data).float()
	label = torch.from_numpy(label).float()
	

	model = LogisticRegression(1, 2)
	criterion = torch.nn.BCELoss(reduction='mean')
	optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

	for epoch in range(epochs):
		for i,j in zip(data,label):
			optimizer.zero_grad()
			outputs = model.forward(i)
			loss = criterion(outputs, j)
			loss.backward()
			optimizer.step()

