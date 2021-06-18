from .fastgaus import render_core_cython as rc 
import os 
import config 
import glob 
from collections import defaultdict
import pickle 
import numpy as np 

class HMapGenerator():
	def __init__(self, data_path):
		instances = sorted(glob.glob(os.path.join(data_path, '*.pkl')))  # we sort here so it's in correct order
		data = defaultdict(list)
		for i in instances:
			i = i.replace('\\','/') # for windows users 

			pts = pickle.load(open(i, 'rb'))[0]
			ones = np.ones([pts.shape[0], pts.shape[1], 1], dtype=np.float32)
			pts = np.concatenate([pts, ones], axis=2)

			vid_idx = int(i.split('/')[-1].split('_')[0])
			data[vid_idx].append(pts)

		self.data = defaultdict(list)
		for k in data:
			self.data[k] = np.stack(data[k], axis=1)

	def get_hmap(self, vid_idx, frame_idx, sigma, h, w, scale):
		all_pts = self.data[vid_idx][frame_idx]
		# print(self.data[vid_idx].shape)
		all_hmaps = [rc.render_heatmap(i * scale *2, h ,w, sigma) for i in all_pts]
		all_hmaps = np.float32(all_hmaps)
		all_hmaps = np.amax(all_hmaps, axis=0)
		return all_hmaps
