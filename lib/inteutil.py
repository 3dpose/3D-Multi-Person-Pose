import numpy as np 
import pickle 
import glob 
import random 
import os 

class InteDataset():
	def __init__(self, bu_path, bu_dep_path, td_path, td_dep_path):
		self.vid_inst = []

		pred_bu = []
		for i in range(20):
			data = pickle.load(open(os.path.join(bu_path, '%d.pkl'%(i+1)), 'rb'))
			data = np.float32(data)
			for j in range(data.shape[1]):
				pred_bu.append(data[:,j])
				self.vid_inst.append([i,j])
		self.pred_bu = pred_bu

		pred_bu_dep = []
		for i in sorted(glob.glob(os.path.join(bu_dep_path,'*.pkl'))):
			dep = pickle.load(open(i, 'rb'))
			pred_bu_dep.append(dep)
		self.pred_bu_dep = pred_bu_dep

		pred_td = []
		for i in range(20):
			data = pickle.load(open(os.path.join(td_path,'%d.pkl'%(i+1)), 'rb'))
			data = np.float32(data)
			for j in range(data.shape[1]):
				pred_td.append(data[:,j])
		self.pred_td = pred_td

		pred_td_dep = []
		for i in sorted(glob.glob(os.path.join(td_dep_path,'*.pkl'))):
			data = pickle.load(open(i, 'rb'))
			pred_td_dep.append(data)
		self.pred_td_dep = pred_td_dep

	def __iter__(self):
		self.pos = 0
		return self 

	def __len__(self):
		return len(self.pred_bu)

	def __next__(self):
		if self.pos>=len(self.pred_bu):
			raise StopIteration
		pred_bu = self.pred_bu[self.pos]
		pred_bu_dep = self.pred_bu_dep[self.pos]
		pred_td = self.pred_td[self.pos]
		pred_td_dep = self.pred_td_dep[self.pos]

		pred_bu = np.float32(pred_bu)
		pred_bu_dep = np.float32(pred_bu_dep)
		pred_td = np.float32(pred_td)
		pred_td_dep = np.float32(pred_td_dep)

		source_pts = np.stack([pred_td, pred_bu], axis=1)
		source_dep = np.stack([pred_td_dep, pred_bu_dep], axis=1)

		num_frames = source_pts.shape[0]
		source_pts = source_pts.reshape([num_frames, -1])
		source_dep = source_dep.reshape([num_frames, -1])
		source_dep = source_dep / 1000

		vid_inst = self.vid_inst[self.pos]

		self.pos += 1

		return source_pts, source_dep, vid_inst

