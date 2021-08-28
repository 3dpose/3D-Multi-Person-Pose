import torch
import numpy as np
from TorchSUL import Model as M

import config
from lib import hmaputil
from lib import btmuputil
from lib.models import transnet
from lib.models import networkbtmup

import glob
import cv2
import pickle
from tqdm import tqdm

import time

if __name__=='__main__':

	start_time = time.time()

	### Stage 1: Do bottom-up estimation
	model = networkbtmup.HR3DNet(config.head_layernum, config.head_chn, config.upsample_layers, config.upsample_chn)
	hmapGen = hmaputil.HMapGenerator('./mupots/est_p2ds/') # detection is applied on 1024 size

	with torch.no_grad():
		# initialize model
		x = torch.zeros(1,3+17,config.inp_size,config.inp_size)
		model(x)
		M.Saver(model).restore('./ckpts/model_btmup/')
		model.eval()
		model.cuda()

		res_all = {}
		for i in range(20):
			print('Seq:',i)
			imgs = sorted(glob.glob('./MultiPersonTestSet/TS%d/*.jpg'%(i+1)))
			buff = []
			for frame_idx,imgname in enumerate(tqdm(sorted(imgs))):
				imgname = imgname.replace('\\','/')  # for windows users
				img = cv2.imread(imgname)
				pts, scores, roots, rels = btmuputil.run_pipeline(img, model, hmap_generator=hmapGen, vid_idx=i, frame_idx=frame_idx)
				pts_final = btmuputil.get_pts3d(pts, roots, rels)
				buff.append([imgname, pts_final, scores])
			res_all[i+1] = buff
		pickle.dump(res_all, open('mupots/MUPOTS_Preds_btmup.pkl','wb'))

	### Stage 2: Transform 14 pts (MUCO format) to 17 pts (mpii format)
	# initialize models
	linear_fit = transnet.LinearModel()
	trans_net = transnet.TransNet()
	x_dumb = torch.from_numpy(np.zeros([2, config.num_pts, 3], dtype=np.float32))
	linear_fit(x_dumb)
	x_dumb = torch.from_numpy(np.zeros([2, config.num_pts*3], dtype=np.float32))
	trans_net(x_dumb)

	# load models
	saver_liner = M.Saver(linear_fit)
	saver_trans = M.Saver(trans_net)
	saver_liner.restore('./ckpts/model_transform/model_linear/')
	saver_trans.restore('./ckpts/model_transform/model_trans/')
	linear_fit.cuda()
	trans_net.cuda()

	# do transformation
	data = pickle.load(open('./mupots/MUPOTS_Preds_btmup.pkl', 'rb'))
	for k in data:
		preds = data[k]
		for frame in tqdm(preds):
			pts = frame[1]
			buff = []
			for i in range(len(pts)):
				pt = pts[i]
				pt_n = btmuputil.normalize_pts(pt, 0)
				pt_n = torch.from_numpy(pt_n.reshape([14,3])).cuda()
				pt_n_refined = linear_fit(pt_n)
				pt_17 = trans_net(pt_n_refined.reshape(1, 14*3)).cpu().detach().numpy()
				pt_17 = pt_17.reshape([17,3])
				buff.append(pt_17)
			frame.append(buff)

	pickle.dump(data, open('./mupots/MUPOTS_Preds_btmup_transformed.pkl', 'wb'))

	end_time = time.time()
	hours, rem = divmod(end_time-start_time, 3600)
	minutes, seconds = divmod(rem, 60)
	print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
