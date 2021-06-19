from lib.models import networkgcn, networktcn 
import torch 
import numpy as np 
from TorchSUL import Model as M 
from tqdm import tqdm
import torch.nn.functional as F 
import pickle 
import glob 
import os 
from collections import defaultdict
import scipy.io as sio 

if __name__=='__main__':
	order = np.int64([10, 8, 14,15,16, 11,12,13, 1,2,3,4,5,6, 0, 7, 9])

	bone_pairs = [[8,9],[9,10], [8,14],[14,15],[15,16], [8,11],[12,13],[11,12], [8,7],[7,0], [4,5],[5,6],[0,4], [0,1],[1,2],[2,3]]
	bone_matrix = np.zeros([16,17], dtype=np.float32)
	for i, pair in enumerate(bone_pairs):
		bone_matrix[i, pair[0]] = -1
		bone_matrix[i, pair[1]] = 1
	bone_matrix_inv = np.linalg.pinv(bone_matrix)
	bone_matrix_inv = torch.from_numpy(bone_matrix_inv)
	bone_matrix = torch.from_numpy(bone_matrix)

	seq_len = 243
	netgcn = networkgcn.TransNet(256, 17)
	nettcn = networktcn.Refine2dNet(17, seq_len)

	# initialize the network with dumb input 
	x_dumb = torch.zeros(2,17,2)
	affb = torch.ones(2,16,16) / 16
	affpts = torch.ones(2,17,17) / 17
	netgcn(x_dumb, affpts, affb, bone_matrix, bone_matrix_inv)
	x_dumb = torch.zeros(2,243, 17*3)
	nettcn(x_dumb)

	# load networks 
	M.Saver(netgcn).restore('./ckpts/model_gcnwild/')
	M.Saver(nettcn).restore('./ckpts/model_tcn/')

	# push to gpu 
	netgcn.cuda()
	netgcn.eval()
	nettcn.cuda()
	nettcn.eval()
	bone_matrix = bone_matrix.cuda()
	bone_matrix_inv = bone_matrix_inv.cuda()

	# create result folder 
	if not os.path.exists('mupots/pred/'):
		os.makedirs('mupots/pred/')

	# run prediction 
	results = defaultdict(list)
	for ptsfile in sorted(glob.glob('mupots/est_p2ds/*.pkl')):
		ptsfile = ptsfile.replace('\\','/') # for windows 
		print(ptsfile)
		p2d, affpts, affb, occmask = pickle.load(open(ptsfile, 'rb'))
		p2d = torch.from_numpy(p2d).cuda() / 1024
		scale = p2d[:,8:9, 1:2] - p2d[:, 0:1, 1:2]
		p2d = p2d / scale
		p2d = p2d - p2d[:,0:1]
		bsize = p2d.shape[0]
		affb = torch.from_numpy(affb).cuda()
		affpts = torch.from_numpy(affpts).cuda()
		occmask = torch.from_numpy(occmask).cuda()
		with torch.no_grad():
			pred = netgcn(p2d, affpts, affb, bone_matrix, bone_matrix_inv)
			pred = pred.unsqueeze(0).unsqueeze(0)
			pred = pred - pred[:,:,:,:1]
			# pred = pred * occmask
			pred = F.pad(pred, (0,0,0,0,seq_len//2, seq_len//2), mode='replicate')
			pred = pred.squeeze()
			pred = nettcn.evaluate(pred)

			# pickle.dump(pred.cpu().numpy(), open(ptsfile.replace('p2ds/', 'pred/'), 'wb'))
			pred = pred.cpu().numpy()
		video_ind = int(ptsfile.split('/')[-1].split('_')[0])
		results[video_ind+1].append(pred)

	for k in results:
		pred = np.stack(results[k], axis=1)
		pred = np.transpose(pred, (0, 1, 3, 2))
		pred = pred[:,:,:,order]
		pickle.dump(pred, open('mupots/pred/%d.pkl'%k, 'wb'))
