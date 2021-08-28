import sys 
sys.path.append('./util/')
from util.mpii_get_joints import mpii_get_joints
import numpy as np 
import copy
import util.matcher 
import util.norm_pose
import util.evaluate
import util.datautil
import util.load_pred
from tqdm import tqdm 
import pickle

_, o1, o2, relevant_labels = mpii_get_joints('relavant')
num_joints = len(o1)

test_annot_base = './MultiPersonTestSet/'

evaluation_mode = 0  # 0 for all, 1 for matched 
safe_traversal_order = [15,16,2,1,17,3,4,5,6,7,8,9,10,11,12,13,14]
safe_traversal_order = [i-1 for i in safe_traversal_order]
sequencewise_per_joint_error = []
sequencewise_undetected_people = []
sequencewise_visibility_mask = []
sequencewise_occlusion_mask = []
sequencewise_annotated_people = []
sequencewise_frames = []
for ts in range(20):
    print('Seq:', ts+1)
    person_ids = []
    oepn_person_ids = list(range(20))
    annots = util.datautil.load_annot(test_annot_base + 'TS%d/annot.mat'%(ts+1))
    occlusions = util.datautil.load_occ(test_annot_base + 'TS%d/occlusion.mat'%(ts+1))
    num_frames = len(annots[0])
    num_person = len(annots)
    undetected_people = 0
    annotated_people = 0 
    pje_idx = 0 
    pje = []
    pjocc = []
    pjvis = []
    sequencewise_frames.append(num_frames)
    for i in range(num_frames):
        valid_annotations = 0;
        for k in range(num_person):
            # print(annots[k][i])
            if annots[k][i]['is_valid']==1:
                valid_annotations += 1
        annotated_people += valid_annotations
        if valid_annotations==0:
            continue

        gt_p2d = []
        gt_p3d = []
        gt_vis = []
        gt_occ = []
        gt_pose_vis = []
        matching_joints = list(range(1,14))
        for k in range(num_person):
            if annots[k][i]['is_valid']==1:
                gt_p2d.append(annots[k][i]['annot2'][:,matching_joints])
                gt_p3d.append(annots[k][i]['annot3'])
                gt_vis.append(np.ones([1,len(matching_joints)]))
                gt_occ.append(occlusions[i][k])
                gt_pose_vis.append(1 - gt_occ[-1])

        # TODO: insert predictions
        # pred_p2d = copy.deepcopy(gt_p2d)
        # pred_p3d = copy.deepcopy(gt_p3d)
        # pred_vis = copy.deepcopy(gt_vis)
        pred_p3d = util.load_pred.get_pred(ts, i)

        matches = util.matcher.match(gt_p3d, pred_p3d, o1, safe_traversal_order[1:])
        for k in range(len(matches)):
            pred_considered = 0
            if matches[k]!=-1:
                gtP = gt_p3d[k] - gt_p3d[k][:,14:15]
                predP = pred_p3d[matches[k]]
                predP = predP - predP[:,14:15]
                predP = util.norm_pose.norm_by_bone_length(predP, gtP, o1, safe_traversal_order[1:])
                # predP = util.norm_pose.procrustes(predP, gtP)
                pred_considered = 1
            else:
                gtP = gt_p3d[k] - gt_p3d[k][:,14:15]
                undetected_people += 1 
                predP = 100000 * np.ones(gtP.shape)
                # print(evaluation_mode)
                if evaluation_mode==0:
                    pred_considered = 1

            if pred_considered == 1:
                errorP = np.sqrt(np.power(predP - gtP, 2).sum(axis=0))
                pje.append(errorP) # num_tested poses 
                pjocc.append(gt_occ[k])
                pjvis.append(gt_vis[k])
    sequencewise_undetected_people.append(undetected_people)
    sequencewise_per_joint_error.append(pje)
    sequencewise_annotated_people.append(annotated_people)
    sequencewise_visibility_mask.append(pjvis)
    sequencewise_occlusion_mask.append(pjocc)

pck_curve_array, pck_array, auc_array = util.evaluate.calculate_multiperson_errors(sequencewise_per_joint_error)
# pickle.dump(pck_curve_array, open('pck_curve_array.pkl', 'wb'))
# pickle.dump(pck_array, open('pck_array.pkl', 'wb'))
# pickle.dump(auc_array, open('auc_array.pkl', 'wb'))
pck_mean = sum([i[-1] for i in pck_array]) / len(pck_array)
print('PCK_MEAN:', pck_mean) 
