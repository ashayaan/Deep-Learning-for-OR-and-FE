# In[104]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder 
import torch.nn as nn
import torch
import pandas as pd


# In[105]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 2
hidden_size1 = 30
hidden_size2 = 100
hidden_size3 = 20
hidden_size4 = 1

num_classes = 1
num_epochs = 300
learning_rate = 0.001


# In[106]:


t = np.arange(1, 101, 1)
n = len(t)
phi_1 = -0.06*t + 3
phi_2 = -0.08*t + 2

r1 = 50 + 0.20*t
r2 = 30 + 0.40*t

x1 = r1 * np.cos(phi_1)
x2 = r2 * np.cos(phi_2)

y1 = r1 * np.sin(phi_1)
y2 = r2 * np.sin(phi_2)

x1 = x1.reshape(n, 1)
x2 = x2.reshape(n, 1)
y1 = y1.reshape(n, 1)
y2 = y2.reshape(n, 1)

x = np.vstack((x1, x2))
y = np.vstack((y1, y2))
labels = np.vstack((np.zeros((n)), np.ones((n))))
labels = labels.reshape((2 * n, 1))

data = np.hstack((x, y, labels))
np.random.shuffle(data)
#print (data)
x_train = data[:,0:-1]
y_train = data[:, -1]
y_train = y_train.reshape((2 * n, 1))

plt.figure(figsize=(12,8))
plt.title('Plot of Curves') 

plt.plot(x1, y1, label = "curve1")
plt.plot(x2, y2, label = "curve2")
plt.xlabel('x') 
plt.ylabel('y')
plt.legend()
plt.savefig("que2.png")
plt.show()


# In[107]:


print (x_train[0])
print (y_train[0])


# In[108]:


class NeuralNet(nn.Module):
    
    def __init__(self, input_size, num_classes):
        
        super(NeuralNet, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.layer1 = nn.Linear(input_size, hidden_size1) 
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.layer3 = nn.Linear(hidden_size2, hidden_size3) 
        self.layer4 = nn.Linear(hidden_size3, hidden_size4)
        
    
    def forward(self, x):
        
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.layer3(out)
        out = self.relu(out)
        out = self.layer4(out)
        print (out)
        #print (out)
        out = self.sigmoid(out)
        #print (out)
        
        return out

model = NeuralNet(input_size, num_classes).to(device)
#print (model)
criterion = nn.BCELoss(reduction = 'mean')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
total_loss_epoch = []
for epoch in range(num_epochs):
    total_loss = 0
    for i in range (2 * n):
        
        x = torch.from_numpy(x_train[i]).float().to(device)
        y = torch.from_numpy(y_train[i]).float().to(device)
    
        outputs = model(x)
        print (outputs)
        loss = criterion(outputs, y)
        #print (outputs)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print (total_loss/(2*n))
    
    print (total_loss/(2*n))
    total_loss_epoch.append(total_loss/(2*n))


# In[109]:


plt.plot(list(range(epoch)), total_loss_epoch[:-1], label = "Decrease of Cost with epoch")
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.legend()
plt.savefig("que2_loss_take2.png")
plt.show()