import numpy as np 
import norm_pose

def match(pose1, pose2, o1=None, trav=None, threshold=250):
    # pose: [2, num_pts]
    # vis: [1, num_pts]
    matches = []
    p2 = np.float32(pose2)
    if o1 is not None:
        p2 = p2 - p2[:,:,14:15]
    for i in range(len(pose1)):
        p1 = np.float32(pose1[i])
        p1 = p1 - p1[:,14:15]
        diffs = []
        for j in range(len(p2)):
            p = p2[j]
            p = norm_pose.procrustes(p, p1)
            diff = np.sqrt(np.power(p-p1,2).sum(axis=0)).mean()
            diffs.append(diff)
        diffs = np.float32(diffs)
        idx = np.argmin(diffs)
        if diffs.min()>threshold:
            matches.append(-1)
        else:
            matches.append(idx)
    return matches
