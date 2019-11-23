import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

batch_size = 1
nz = 100
nc= 3
ngf = 512
ndf = 64
lr = 0.0002

class Generator(nn.Module):
	def __init__(self):
		super(Generator,self).__init__()
		self.generator = nn.Sequential(
			
			nn.ConvTranspose2d( nz, ngf * 8, 8, 1, 0, bias=False),
			nn.BatchNorm2d(ngf * 8),
			nn.ReLU(True),
			
			nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 4, 0, bias=False),
			nn.BatchNorm2d(ngf * 4),
			nn.ReLU(True),
			
			nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 4, 0, bias=False),
			nn.BatchNorm2d(ngf * 2),
			nn.ReLU(True),
			
			nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf),
			nn.ReLU(True),
			
			nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
			nn.ReLU(True)
		)

	def forward(self,data):
		return self.generator(data)

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator,self).__init__()
		self.discriminator = nn.Sequential(
				
				nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
				nn.LeakyReLU(0.2, inplace=True),
				
				nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
				nn.BatchNorm2d(ndf * 2),
				nn.LeakyReLU(0.2, inplace=True),
				
				nn.Conv2d(ndf * 2, ndf * 4, 4, 4, 0, bias=False),
				nn.BatchNorm2d(ndf * 4),
				nn.LeakyReLU(0.2, inplace=True),
				
				nn.Conv2d(ndf * 4, ndf * 8, 4, 4, 0, bias=False),
				nn.BatchNorm2d(ndf * 8),
				nn.LeakyReLU(0.2, inplace=True),
				
				nn.Conv2d(ndf * 8, 1, 8, 1, 0, bias=False),
				nn.Sigmoid()
			)

	def forward(self,data):
		return self.discriminator(data)

def trainingGAN(data_generator,data_discriminator):
	criterion = nn.BCELoss()
	real_label = 1
	fake_label = 0

	gen = Generator()
	dis = Discriminator()

	print (gen)
	print (dis)

	optimizerD = optim.Adam(dis.parameters(),lr=lr)
	optimizerG = optim.Adam(gen.parameters(),lr=lr)


	for epoch in range(1):

		dis.zero_grad()
		label_real = torch.tensor(real_label,dtype=torch.float32)

		output = dis.forward(data_discriminator)
		errD_real = criterion(output, label_real)
		errD_real.backward()
		D_x = output.mean().item()

		fake_data = gen.forward(data_generator)
		label_fake = torch.tensor(fake_label,dtype=torch.float32)

		output = dis.forward(fake_data.detach()).view(-1)
		errD_fake = criterion(output, label_fake)
		errD_fake.backward()

		D_G_z1 = output.mean().item()
		errD = errD_real + errD_fake
		optimizerD.step()

		gen.zero_grad()
		output = dis.forward(fake_data).view(-1)
		errG = criterion(output, label_real)
		errG.backward()
		D_G_z2 = output.mean().item()
		optimizerG.step()


		print ('Loss_D: %.4f Loss_G: %.4f'%(errD,errG))




if __name__ == '__main__':
	data_generator = torch.randn(1, nz, 1, 1)
	data_discriminator = torch.randn(1,3,512,512)

	trainingGAN(data_generator,data_discriminator)
