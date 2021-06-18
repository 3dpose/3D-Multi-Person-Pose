from torch.nn.parameter import Parameter
import torch.nn.init as init 
from TorchSUL import Model as M 
import torch 

class TransNet(M.Model):
	def initialize(self):
		self.f3 = M.Dense(3*17)

	def forward(self, x):
		x = self.f3(x)
		return x 

class LinearModel(M.Model):
	def initialize(self):
		self.weight = Parameter(torch.Tensor(3))
		self.bias = Parameter(torch.Tensor(3))
		init.normal_(self.weight, std=0.001)
		init.zeros_(self.bias)

	def forward(self, x):
		x = self.weight * x 
		x = x + self.bias
		return x 
