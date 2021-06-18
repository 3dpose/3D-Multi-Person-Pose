from collections import defaultdict
import numpy as np
import pickle
import glob
import os
from util.norm_pose import procrustes
from tqdm import tqdm 

class PoseMatcher():
    def __init__(self, top_down_path, btm_up_path):
        self.top_down_path = top_down_path
        self.btm_up_path = btm_up_path

    def _best_match(self, ref, targets):
        def OKS(p1, p2):
            sigma = 0.05
            selected_idx = np.int64([14, 8,9, 11,12, 5,6, 2,3])
            p1_selected = p1[:2, selected_idx]
            p2_selected = p2[:2, selected_idx]
            p1_selected -= p1_selected[:,0:1]
            p2_selected -= p2_selected[:,0:1]
            dist = np.square(p1_selected - p2_selected).sum(axis=0)
            result = np.exp(- dist / (2*sigma**2)).mean()
            return result
        max_idx = 0
        max_oks = 0
        max_pts = 0
        for i in range(len(targets)):
            aligned_target = procrustes(targets[i], ref)
            oks = OKS(ref, aligned_target)
            if oks>max_oks:
                max_oks = oks
                max_idx = i
                max_pts = aligned_target
        return max_pts, max_idx

    def match(self, pts_out_path, dep_out_path, gt_dep_path):
        # create directory 
        if not os.path.exists(pts_out_path):
            os.makedirs(pts_out_path)
        if not os.path.exists(dep_out_path):
            os.makedirs(dep_out_path)
        # Load depth gt for alignment 
        depth_gts = defaultdict(list)
        for depthfile in sorted(glob.glob(os.path.join(gt_dep_path,'*.pkl'))):
            depthfile = depthfile.replace('\\','/')  # for windows users 
            video_ind = int(depthfile.split('/')[-1].split('_')[0])
            depth_gt = pickle.load(open(depthfile , 'rb'))
            depth_gts[video_ind].append(depth_gt)
        
        # Load predictions 
        bu_estimations = pickle.load(open(self.btm_up_path, 'rb'))

        # we sort here so it's in correct order, to prevent some linux os produce incorrect order 
        vid_results = sorted(glob.glob(os.path.join(self.top_down_path, '*.pkl')))

        for v in tqdm(vid_results):
            v = v.replace('\\', '/')  # for windows users
            pts = pickle.load(open(v, 'rb'))
            vid_idx = int(v.split('/')[-1].split('.')[0])

            # # match 
            results = []
            depths = []
            for frame in range(len(pts)):
                bu_pts = bu_estimations[vid_idx][frame][3]
                bu_pts = np.float32(bu_pts).transpose([0,2,1])
                bu_depths = np.float32(bu_estimations[vid_idx][frame][1])

                buff_p = []
                buff_d = []
                for p in pts[frame]:
                    p_aligned, idx = self._best_match(p , bu_pts)
                    buff_p.append(p_aligned)
                    buff_d.append(bu_depths[idx])

                results.append(buff_p)
                depths.append(buff_d)
            
            pickle.dump(results, open(os.path.join(pts_out_path,'%d.pkl'%vid_idx), 'wb'))
            
            # align the depth 
            depths = np.float32(depths)[:,:,0,2].transpose([1,0])   #[Num_inst, num_frame]
            num_inst = depths.shape[0]
            depth_gt = np.float32(depth_gts[vid_idx-1])
            
            depths = depths.reshape(-1)
            depth_gt = depth_gt.reshape(-1)
            depths = procrustes(depths[None, ...], depth_gt[None, ...])[0]
            depths = depths.reshape([num_inst, -1])
            for i in range(depths.shape[0]):
                pickle.dump(depths[i], open(os.path.join(dep_out_path,'%02d_%02d.pkl'%(vid_idx-1, i)),'wb'))
