import numpy as np 

def norm_by_bone_length(pred, gt, o1, trav):
    mapped_pose = pred.copy()

    for i in range(len(trav)):
        idx = trav[i]
        gt_len = np.linalg.norm(gt[:,idx] - gt[:,o1[i]])
        pred_vec = pred[:, idx] - pred[:,o1[i]]
        pred_len = np.linalg.norm(pred_vec)
        mapped_pose[:, idx] = mapped_pose[:, o1[i]] + pred_vec * gt_len / pred_len
    return mapped_pose

def procrustes(predicted, target):
    predicted = predicted.T 
    target = target.T
    predicted = predicted[None, ...]
    target = target[None, ...]

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    
    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t
    return predicted_aligned[0].T

