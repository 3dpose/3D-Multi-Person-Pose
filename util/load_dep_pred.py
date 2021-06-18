import numpy as np 
import pickle 

data_dict = {}
gt_dict = {}
def get_pred_gt(seq_idx, inst_idx, frame_idx):
    global data_dict
    if not seq_idx in data_dict:
        data = pickle.load(open('mupots/pred_dep_inte/%02d_%02d.pkl'%(seq_idx, inst_idx), 'rb'))
        data_dict[(seq_idx, inst_idx)] = np.float32(data)
        gt = pickle.load(open('mupots/depths/%02d_%02d.pkl'%(seq_idx, inst_idx), 'rb'))
        gt_dict[(seq_idx, inst_idx)] = np.float32(gt)
    data = data_dict[(seq_idx, inst_idx)]
    gt = gt_dict[(seq_idx, inst_idx)]
    return data[frame_idx], gt[frame_idx]
