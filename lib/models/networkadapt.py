import numpy as np 
import torch 
import TorchSUL.Model as M 

class AdaptNet(M.Model):
    def initialize(self):
        self.l0 = M.Dense(512, activation=M.PARAM_GELU)
        self.l1 = M.Dense(512, activation=M.PARAM_GELU)
        self.l2 = M.Dense(9)
        self.l3 = M.Dense(17*3)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).cuda()
        x = x.reshape(-1, 17*3)
        a = self.l0(x)
        a = self.l1(a)
        R = self.l2(a)
        R = R.reshape(-1, 3, 3)
        x = x.reshape(-1, 3, 17)
        x_rotated = x = torch.einsum('ijk,ijl->ikl', R, x)
        x = x.reshape(-1, 17*3)
        x = self.l3(x)
        x = x.reshape(-1, 3, 17)
        return x

