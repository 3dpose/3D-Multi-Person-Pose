import numpy as np 
import torch 
import torch.nn.functional as F 
from TorchSUL import Model as M 
from torch.nn.parameter import Parameter
import torch.nn.init as init 

class PropLayer(M.Model):
	def initialize(self, outdim, usebias=True):
		self.outdim = outdim
		self.act = torch.nn.ReLU()
		self.act2 = torch.nn.ReLU()
		self.usebias = usebias

	def build(self, *inp):
		# inp: [Bsize, num_pts, 2]
		num_pts = inp[0].shape[1]
		indim = inp[0].shape[2]
		self.weight = Parameter(torch.Tensor(num_pts, indim, self.outdim))
		self.weight2 = Parameter(torch.Tensor(num_pts, self.outdim, self.outdim))
		init.kaiming_uniform_(self.weight, a=np.sqrt(5))
		init.kaiming_uniform_(self.weight2, a=np.sqrt(5))
		if self.usebias:
			print('initialize bias')
			self.bias = Parameter(torch.Tensor(num_pts, self.outdim)) 
			self.bias2 = Parameter(torch.Tensor(num_pts, self.outdim)) 
			init.uniform_(self.bias, -0.1, 0.1)
			init.uniform_(self.bias2, -0.1, 0.1)

	def forward(self, inp, aff=None, act=True):
		if aff is not None:
			# propagate the keypoints 
			x = torch.einsum('ikl,ijk->ijl', inp, aff)
		else:
			x = inp 

		x = torch.einsum('ijk,jkl->ijl', x, self.weight)
		if self.usebias:
			x = x + self.bias
		if act:
			x = self.act(x)
		# x = F.dropout(x, 0.25, self.training, False)

		x = torch.einsum('ijk,jkl->ijl', x, self.weight2)
		if self.usebias:
			x = x + self.bias2
		if act:
			x = self.act2(x)
		#x = F.dropout(x, 0.25, self.training, False)

		if aff is not None:
			x = torch.cat([inp, x], dim=-1)
		return x 

class TransNet(M.Model):
	def initialize(self, outdim, num_pts):
		self.num_pts = num_pts
		self.c1 = PropLayer(outdim)
		self.c2 = PropLayer(outdim)
		self.c3 = PropLayer(outdim)

		self.b2 = PropLayer(outdim)
		self.b3 = PropLayer(outdim)

		self.c8 = PropLayer(outdim)
		self.c9 = PropLayer(3)

	def forward(self, x, aff, aff_bone, inc, inc_inv):
		x = feat = self.c1(x)
		x = self.c2(x, aff)
		x = self.c3(x, aff)

		feat = torch.einsum('ijk,lj->ilk', feat, inc)
		feat = self.b2(feat, aff_bone)
		feat = self.b3(feat, aff_bone)
		feat = torch.einsum('ijk,lj->ilk', feat, inc_inv)
		x = torch.cat([x, feat], dim=-1)
		
		x = self.c8(x)
		x = self.c9(x, act=False)
		# print(x.shape)
		x = x.reshape(-1, self.num_pts, 3)
		return x 
