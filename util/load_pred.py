import scipy.io as sio 
import numpy as np 
import pickle 
from lib.models import networkadapt 
import torch 
import TorchSUL.Model as M 

net = networkadapt.AdaptNet()
pts_dumb = torch.zeros(2, 17*3)
net(pts_dumb)
M.Saver(net).restore('./ckpts/model_adapt/')
net.cuda()

data_dict = {}
def get_pred(seq_idx, frame_idx):
    global data_dict
    seq_idx = seq_idx + 1 
    if not seq_idx in data_dict:
        data = pickle.load(open('mupots/pred_inte/%d.pkl'%seq_idx, 'rb'))
        data_dict[seq_idx] = np.float32(data)
    pts = data_dict[seq_idx][frame_idx]
    with torch.no_grad():
        pts = net(pts).cpu().numpy()
    return pts
