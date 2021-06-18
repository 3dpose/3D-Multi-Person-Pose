from lib.models import networktcn 
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
import util.norm_pose

if __name__=='__main__':
	seq_len = 243
	nettcn = networktcn.Refine2dNet(17, seq_len, input_dimension=2, output_dimension=1, output_pts=1)
	x_dumb = torch.zeros(2,243, 17*2)
	nettcn(x_dumb)
	M.Saver(nettcn).restore('./ckpts/model_root/')
	nettcn.cuda()
	nettcn.eval()

	# create result folder 
	if not os.path.exists('mupots/pred_dep/'):
		os.makedirs('mupots/pred_dep/')

	results = defaultdict(list)
	gts = defaultdict(list)
	for ptsfile in sorted(glob.glob('mupots/est_p2ds/*.pkl')):
		ptsfile = ptsfile.replace('\\','/') # for windows 
		print(ptsfile)
		p2d, affpts, affb, occmask = pickle.load(open(ptsfile, 'rb'))
		p2d = torch.from_numpy(p2d).cuda() / 915

		with torch.no_grad():
			p2d = p2d.unsqueeze(0).unsqueeze(0)
			p2d = F.pad(p2d, (0,0,0,0,seq_len//2, seq_len//2), mode='replicate')
			p2d = p2d.squeeze()
			pred = nettcn.evaluate(p2d)
			pred = pred.cpu().numpy()

		# do pa alignment
		video_ind = int(ptsfile.split('/')[-1].split('_')[0])
		depth_gt = pickle.load(open(ptsfile.replace('est_p2ds', 'depths') , 'rb'))
		results[video_ind].append(pred)
		gts[video_ind].append(depth_gt)

	for key in results:
		preds = results[key]
		depth_gt = gts[key]

		preds_cat = np.concatenate(preds)
		depth_gt_cat = np.concatenate(depth_gt)

		pred_aligned = util.norm_pose.procrustes(preds_cat[None, ...], depth_gt_cat[None, ...])[0]
		pred_aligned = pred_aligned.reshape(len(preds), -1)

		# save result
		for i in range(len(preds)):
			pickle.dump(pred_aligned[i], open('mupots/pred_dep/%02d_%02d.pkl'%(key, i), 'wb'))
