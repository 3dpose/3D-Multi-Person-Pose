import scipy.io as sio 
import numpy as np 
import pickle 

data_dict = {}
def get_pred(seq_idx, frame_idx):
    global data_dict
    seq_idx = seq_idx + 1 
    if not seq_idx in data_dict:
        # data = sio.loadmat('./results/%d.mat'%seq_idx)['preds']
        data = pickle.load(open('mupots/pred_inte/%d.pkl'%seq_idx, 'rb'))
        data_dict[seq_idx] = np.float32(data)
    data = data_dict[seq_idx]
    return data[frame_idx]
