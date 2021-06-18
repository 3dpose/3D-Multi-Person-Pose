import os 
import torch
import pickle 
import numpy as np 

from lib import inteutil
from lib import posematcher
from lib.models import networkinte

from tqdm import tqdm 
from TorchSUL import Model as M 
from collections import defaultdict 

if __name__=='__main__':
	## step 1: match the poses 
	print('Matching poses from two branches...')
	matcher = posematcher.PoseMatcher(top_down_path='./mupots/pred/', 
								btm_up_path='./mupots/MUPOTS_Preds_btmup_transformed.pkl')
	matcher.match(pts_out_path='./mupots/pred_bu/', dep_out_path='./mupots/pred_dep_bu/', 
				gt_dep_path='./mupots/depths/')

	## step 2: infer the integrated results 
	print('Inferring the integrated poses...')
	# create data loader 
	data = inteutil.InteDataset(bu_path='./mupots/pred_bu/', bu_dep_path='./mupots/pred_dep_bu/',
								td_path='./mupots/pred/', td_dep_path='./mupots/pred_dep/')
	# initialize the network
	net = networkinte.IntegrationNet()
	pts_dumb = torch.zeros(2, 102)
	dep_dumb = torch.zeros(2, 2)
	net(pts_dumb, dep_dumb)
	M.Saver(net).restore('./ckpts/model_inte/')
	net.cuda()

	# create paths 
	if not os.path.exists('./mupots/pred_inte/'):
		os.makedirs('./mupots/pred_inte/')
	if not os.path.exists('./mupots/pred_dep_inte/'):
		os.makedirs('./mupots/pred_dep_inte/')

	with torch.no_grad():
		all_pts = defaultdict(list)
		for src_pts,src_dep,vid_inst in tqdm(data):
			src_pts = torch.from_numpy(src_pts).cuda()
			src_dep = torch.from_numpy(src_dep).cuda()
			res_pts, res_dep = net(src_pts, src_dep)
			res_pts = res_pts.cpu().numpy()
			res_dep = res_dep.squeeze().cpu().numpy() * 1000  # the depth is scaled 1000

			# save results 
			i,j = vid_inst
			all_pts[i].insert(j, res_pts)
			pickle.dump(res_dep, open('./mupots/pred_dep_inte/%02d_%02d.pkl'%(i,j), 'wb'))

		for k in all_pts:
			result = np.stack(all_pts[k], axis=1)
			pickle.dump(result, open('./mupots/pred_inte/%d.pkl'%(k+1), 'wb'))
