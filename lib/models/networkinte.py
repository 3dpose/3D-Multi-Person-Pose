import torch 
import numpy as np 
import TorchSUL.Model as M 

class IntegrationNet(M.Model):
	def initialize(self):
		self.fc1 = M.Dense(512, activation=M.PARAM_RELU)
		self.fc2 = M.Dense(512, activation=M.PARAM_RELU)
		self.fc3 = M.Dense(512, activation=M.PARAM_RELU)
		self.fc4 = M.Dense(2, activation=M.PARAM_TANH)

	def forward(self, pts, depths):
		x = torch.cat([pts, depths], dim=1)
		x = self.fc1(x)
		x = self.fc2(x)
		x = self.fc3(x)
		x = self.fc4(x)

		bsize = pts.shape[0]
		pts = pts.reshape(bsize, 2, -1)
		w_pt = x[:,0:1]
		pts = w_pt * pts[:,0] + (1 - w_pt) * pts[:,1]
		pts = pts.reshape(bsize, 3, 17)

		w_dep = x[:,1:2]
		depths = w_dep * depths[:,0:1] + (1 - w_dep) * depths[:,1:2]

		return pts, depths

