import numpy as np 
import cv2 
import torch 
import torch.nn.functional as F 
import config 
import pickle 
from TorchSUL import Model as M 

from munkres import Munkres

def transform_preds(coords, center, scale, output_size):
	# target_coords = np.zeros(coords.shape)
	target_coords = coords.copy()
	trans = get_affine_transform(center, scale, 0, output_size, inv=1)
	for p in range(coords.shape[0]):
		target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
	return target_coords

def get_affine_transform(center,
						 scale,
						 rot,
						 output_size,
						 shift=np.array([0, 0], dtype=np.float32),
						 inv=0):
	if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
		print(scale)
		scale = np.array([scale, scale])

	scale_tmp = scale * 200.0
	src_w = scale_tmp[0]
	dst_w = output_size[0]
	dst_h = output_size[1]

	rot_rad = np.pi * rot / 180
	src_dir = get_dir([0, src_w * -0.5], rot_rad)
	dst_dir = np.array([0, dst_w * -0.5], np.float32)

	src = np.zeros((3, 2), dtype=np.float32)
	dst = np.zeros((3, 2), dtype=np.float32)
	src[0, :] = center + scale_tmp * shift
	src[1, :] = center + src_dir + scale_tmp * shift
	dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
	dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

	src[2:, :] = get_3rd_point(src[0, :], src[1, :])
	dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

	if inv:
		trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
	else:
		trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

	return trans

def affine_transform(pt, t):
	new_pt = np.array([pt[0], pt[1], 1.]).T
	new_pt = np.dot(t, new_pt)
	return new_pt[:2]

def get_3rd_point(a, b):
	direct = a - b
	return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_dir(src_point, rot_rad):
	sn, cs = np.sin(rot_rad), np.cos(rot_rad)
	src_result = [0, 0]
	src_result[0] = src_point[0] * cs - src_point[1] * sn
	src_result[1] = src_point[0] * sn + src_point[1] * cs
	return src_result

def resize(image, input_size):
	h, w, _ = image.shape

	center = np.array([int(w/2.0+0.5), int(h/2.0+0.5)])
	if w < h:
		w_resized = input_size
		h_resized = int((input_size / w * h + 63) // 64 * 64)
		scale_w = w / 200.0
		scale_h = h_resized / w_resized * w / 200.0
	else:
		h_resized = input_size
		w_resized = int((input_size / h * w + 63) // 64 * 64)
		scale_h = h / 200.0
		scale_w = w_resized / h_resized * h / 200.0

	scale = np.array([scale_w, scale_h])
	trans = get_affine_transform(center, scale, 0, (w_resized, h_resized))

	image_resized = cv2.warpAffine(
		image,
		trans,
		(int(w_resized), int(h_resized))
	)
	return image_resized, center, scale

def get_multi_scale_size(image, input_size, current_scale, min_scale):
	h, w, _ = image.shape
	center = np.array([int(w / 2.0 + 0.5), int(h / 2.0 + 0.5)])

	# calculate the size for min_scale
	min_input_size = int((min_scale * input_size + 63)//64 * 64)
	if w < h:
		w_resized = int(min_input_size * current_scale / min_scale)
		h_resized = int(
			int((min_input_size/w*h+63)//64*64)*current_scale/min_scale
		)
		scale_w = w / 200.0
		scale_h = h_resized / w_resized * w / 200.0
	else:
		h_resized = int(min_input_size * current_scale / min_scale)
		w_resized = int(
			int((min_input_size/h*w+63)//64*64)*current_scale/min_scale
		)
		scale_h = h / 200.0
		scale_w = w_resized / h_resized * h / 200.0

	return (w_resized, h_resized), center, np.array([scale_w, scale_h])

def resize_align_multi_scale(image, input_size, current_scale, min_scale):
	size_resized, center, scale = get_multi_scale_size(
		image, input_size, current_scale, min_scale
	)
	trans = get_affine_transform(center, scale, 0, size_resized)

	image_resized = cv2.warpAffine(
		image,
		trans,
		size_resized
		# (int(w_resized), int(h_resized))
	)
	return image_resized, center, scale

def get_final_preds(grouped_joints, center, scale, heatmap_size):
	final_results = []
	for person in grouped_joints[0]:
		joints = np.zeros((person.shape[0], 3))
		joints = transform_preds(person, center, scale, heatmap_size)
		final_results.append(joints)
	return final_results

def pre_process(img):
	img = np.float32(img)
	img = img / 255 
	# img = img[:,:,::-1]
	img = img - np.float32([0.485, 0.456, 0.406])
	img = img / np.float32([0.229, 0.224, 0.225])

	img = np.transpose(img, [2,0,1])
	# img = torch.from_numpy(img)
	return img 

def rescale_img2(img):
	h,w = img.shape[:2]
	if h<w:
		hh = htarget = int(config.inp_size) 
		wtarget = int((w * config.inp_size / h + 63) // 64 * 64)
		ww = int(w * hh / h)
	else:
		ww = wtarget = int(config.inp_size)
		htarget = int((h * config.inp_size / w + 63) // 64 * 64)
		hh = int(h * ww / w)
	scaleh = hh / h 
	scalew = ww / w 
	padh = (htarget - hh) // 2 
	padw = (wtarget - ww) // 2 
	canvas = np.zeros([htarget, wtarget, 3], dtype=np.uint8)
	img = cv2.resize(img, (ww, hh))
	canvas[padh:padh+hh, padw:padw+ww] = img 
	return canvas, [padw, padh, scaleh, scalew]

def compare(arr, pkl):
	data = pickle.load(open(pkl, 'rb'))
	diff = np.float32(arr) - np.float32(data) 
	diff = np.sum(np.abs(diff))
	print('DIFF', diff, arr.shape, data.shape, arr.dtype, data.dtype)

def process_img(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img, center, scale = resize_align_multi_scale(img, config.inp_size, 1, 1)
	# print(center)
	# print(scale)

	img = pre_process(img)
	img = img[None,...]
	# compare(img, 'image.pkl')
	img = torch.from_numpy(img).cuda()
	return img, center, scale

def get_heatmap(img, model, inputhmap=None):
	hmap = 0 
	tags = []
	roots = 0
	rels = 0
	if inputhmap is None:
		outputs = model(img)
	else:
		inputhmap = torch.from_numpy(inputhmap[None,...]).cuda()
		# print(img.shape, inputhmap.shape)
		inp = torch.cat([img, inputhmap], dim=1)
		outputs = model(inp)
	for output in outputs:
		output = F.interpolate(output, size=(outputs[-1].shape[2], outputs[-1].shape[3]), mode='bilinear')
		hmap = hmap + output[:,:config.num_pts]
		tags.append(output[:,config.num_pts:config.num_pts*2])
		roots = roots + output[:,config.num_pts*2:config.num_pts*2+1]
		rels = rels + output[:, config.num_pts*2+1:]
	hmap = hmap / len(outputs)
	roots = roots / len(outputs)
	rels = rels / len(outputs)

	hmap_flip = 0
	roots_flip = 0 
	rels_flip = 0
	# flip_index = [0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15]
	flip_index = [0, 4,5,6, 1,2,3, 7, 11,12,13, 8,9,10]
	if inputhmap is None:
		outputs_flip = model(torch.flip(img, [3]))
	else:
		outputs_flip = model(torch.flip(inp, [3]))
	for output in outputs_flip:
		outout = F.interpolate(output, size=(outputs_flip[-1].shape[2], outputs[-1].shape[3]), mode='bilinear')
		output = torch.flip(output, [3])
		hmap_flip = hmap_flip + output[:,:config.num_pts][:, flip_index]
		tags.append(output[:,config.num_pts:config.num_pts*2][:, flip_index])
		buff = output[:,config.num_pts*2:][:, flip_index]
		roots_flip = roots_flip + buff[:,0:1]
		rels_flip = rels_flip + buff[:,1:]
	hmap_flip = hmap_flip / len(outputs_flip)
	roots_flip = roots_flip / len(outputs_flip)
	rels_flip = rels_flip / len(outputs_flip)

	h, w = img.shape[2], img.shape[3]
	hmap = F.interpolate(hmap, size=(h,w), mode='bilinear')
	hmap_flip = F.interpolate(hmap_flip, size=(h,w), mode='bilinear')
	roots = F.interpolate(roots, size=(h,w), mode='bilinear')
	roots_flip = F.interpolate(roots_flip, size=(h,w), mode='bilinear')
	rels = F.interpolate(rels, size=(h,w), mode='bilinear')
	rels_flip = F.interpolate(rels_flip, size=(h,w), mode='bilinear')
	tags = [F.interpolate(t, size=(h,w), mode='bilinear') for t in tags]
	# print(hmap.shape)
	# hmap_final = (hmap + hmap_flip) / 2.0
	# roots_final = (roots + roots_flip) / 2.0
	# rels_final = (rels + rels_flip) / 2.0
	hmap_final = hmap
	roots_final = roots
	rels_final = rels
	# hmap_final = hmap_flip
	# roots_final = roots_flip
	# rels_final = rels_flip
	tags = [t.unsqueeze(4) for t in tags]
	tags = torch.cat(tags, dim=4)
	return hmap_final, tags, roots_final, rels_final

def top_k(hmap, tags):
	pool = M.MaxPool2D(config.nms_kernel)
	hmap_m = pool(hmap)
	hmap_m = torch.eq(hmap_m, hmap).float()
	hmap = hmap_m * hmap 

	bsize, num_pts, h, w = hmap.shape
	hmap = hmap.view(bsize, num_pts, h * w)
	tags = tags.view(bsize, num_pts, h * w, -1)
	val, ind = hmap.topk(config.max_inst, dim=2)
	ind_expanded = ind.unsqueeze(-1).expand(-1,-1,-1,tags.shape[-1])
	tagk = torch.gather(tags, 2, ind_expanded)

	x = ind%w 
	y = (ind/w).long()
	indk = torch.stack((x,y), dim=3)
	vals, indk, tagk = val[0], indk[0], tagk[0]
	return vals, indk, tagk

def hungarian(scores):
	if not isinstance(scores, np.ndarray):
		scores = scores.cpu().detach().numpy()
	m = Munkres()
	res = m.compute(scores)
	res = np.int32(res)
	return res 

def grouping(vals, indk, tagk):
	joint_order = [0,1,4,2,5,3,6,7,8,11,9,12,10,13]
	# res = torch.zeros(config.num_pts, 3+tagk.shape[2]).float()
	joint_dict = {}
	tag_dict = {}
	for i,idx in enumerate(joint_order):
		tags = tagk[idx]
		val = vals[idx]
		mask = torch.where(val>config.hmap_thresh)[0]
		if len(mask)==0:
			continue 
		tags = tags[mask] # [m, 2]
		val = val[mask] # [m]
		ind = indk[idx, mask] #[m, 2]
		if i==0 or len(joint_dict)==0:
			for t,v,ii in zip(tags, val, ind):
				key = t[0]
				value = torch.zeros(config.num_pts, 3+tagk.shape[2]).float().cuda()
				value[idx, :2] = ii 
				value[idx, 2] = v 
				value[idx, 3:] = t 
				joint_dict[key] = value
				tag_dict[key] = [t]
		else:
			grouped_keys = list(joint_dict.keys())
			grouped_tags = torch.stack([torch.mean(torch.stack(tag_dict[kk]),dim=0) for kk in grouped_keys], dim=0) # [g, 2]
			diff = tags.unsqueeze(1) - grouped_tags.unsqueeze(0)
			diff_norm = torch.norm(diff, dim=2) # [m,g]
			diff_normed2 = torch.round(diff_norm) * 100 - val.unsqueeze(1)
			num_added, num_grouped = diff.shape[0], diff.shape[1]
			if num_added > num_grouped:
				diff_normed2 = torch.cat([diff_normed2, torch.zeros(num_added, num_added-num_grouped).cuda()], dim=1)
			pairs = hungarian(diff_normed2)
			diffshape = diff_norm.shape 
			for row,col in pairs:
				is_matched = False
				if (row<num_added and col<num_grouped):
					if diff_norm[row,col]<config.tag_thresh:
						key = grouped_keys[col]
						value = joint_dict[key][idx]
						value[:2] = ind[row]
						value[2] = val[row]
						value[3:] = tags[row]
						tag_dict[key].append(tags[row])
						is_matched = True
				if not is_matched:
					key = tags[row][0]
					value = torch.zeros(config.num_pts, 3+tagk.shape[2]).float().cuda()
					value[idx,:2] = ind[row]
					value[idx,2] = val[row]
					value[idx,3:] = tags[row]
					joint_dict[key] = value 
					tag_dict[key] = [tags[row]]

	ans = [joint_dict[i] for i in joint_dict]
	if len(ans)==0:
		return [] 
	ans = torch.stack(ans, dim=0)
	return ans 

def adjust(pts_init, hmap):
	pts_init = pts_init.clone()
	for inst_id, inst in enumerate(pts_init):
		for p, pts in enumerate(inst):
			if pts[2]>0:
				x,y = pts[0:2]
				xx,yy = x.long(), y.long()
				h = hmap[0,p]
				if h[yy,min(xx+1,h.shape[1]-1)]>h[yy,max(xx-1,0)]:
					x += 0.25 
				else:
					x -= 0.25 
				if h[min(yy+1,h.shape[0]-1),xx]>h[max(yy-1,0),xx]:
					y += 0.25 
				else:
					y -= 0.25 
				pts[0] = x+0.5 
				pts[1] = y+0.5

	scores = pts_init[:,:,2].mean(dim=1) 
	return pts_init, scores 

def refine(pts_adjusted, hmap, tags):
	res = []
	for i in range(pts_adjusted.shape[0]):
		pts = pts_adjusted[i]
		init_tag = []
		for p in range(config.num_pts):
			if pts[p][2]>0:
				x,y = pts[p][0].long(), pts[p][1].long()
				init_tag.append(tags[0,p,y,x])
		init_tag = torch.mean(torch.stack(init_tag, dim=0), dim=0)

		buff = []
		for p in range(config.num_pts):
			dist = tags[0,p] - init_tag 
			dist = torch.sqrt(torch.sum(torch.pow(dist, 2), dim=2)) # [h,w]
			conf = hmap[0,p] - torch.round(dist / config.tag_distance)
			confmax = conf.max()
			idx = torch.argmax(conf)
			idx = idx.cpu().detach().numpy()
			idx = int(idx)
			x = idx % int(conf.shape[1])
			y = idx // int(conf.shape[1])
			xx = int(x)
			yy = int(y)
			v = hmap[0,p,yy,xx].cpu().detach().numpy()
			x += 0.5 
			y += 0.5 
			if hmap[0,p,yy,min(xx+1,conf.shape[1]-1)]>hmap[0,p,yy,max(xx-1,0)]:
				x += 0.25 
			else:
				x -= 0.25 
			if hmap[0,p,min(yy+1,conf.shape[0]-1),xx]>hmap[0,p,max(yy-1,0),xx]:
				y += 0.25
			else:
				y -= 0.25
			buff.append([x,y,v])
		res.append(buff)
	res = np.float32(res)
	original = pts_adjusted.cpu().detach().numpy()
	for i in range(original.shape[0]):
		# pts = original[i]
		for p in range(original.shape[1]):
			if res[i,p,2]>0 and original[i,p,2]==0:
				original[i,p,:3] = res[i,p,:3]
	return original 

def run_pipeline(inputimg, model, hmap_generator=None, vid_idx=None, frame_idx=None):
	img, center, scale = process_img(inputimg)
	if hmap_generator is None:
		inputhmap = None
	else:
		scale_hmap = img.shape[2] / inputimg.shape[0] 
		inputhmap = hmap_generator.get_hmap(vid_idx, frame_idx, config.base_sigma, img.shape[2], img.shape[3], scale_hmap)

	hmap, tags, roots, rels = get_heatmap(img, model, inputhmap=inputhmap)
	vals, indk, tagk = top_k(hmap, tags)
	pts_init = grouping(vals, indk, tagk)
	if len(pts_init)==0:
		return [], [], [], []
	pts_adjusted, scores = adjust(pts_init, hmap)
	pts_refined = refine(pts_adjusted, hmap, tags)
	pts_final = get_final_preds([pts_refined], center, scale, [hmap.size(3), hmap.size(2)])
	return pts_final, scores, roots, rels

def get_pts3d(pts, roots, rels):
	roots = roots.cpu().detach().numpy()
	rels = rels.cpu().detach().numpy()
	results = []
	for p in pts:
		buff = np.zeros([config.num_pts, 3], dtype=np.float32)
		x = int(p[0,0] / 4)
		y = int(p[0,1] / 4)
		dep = roots[0,0,y,x]
		buff[0,2] = dep
		for i in range(config.num_pts - 1):
			x = int(p[i+1, 0] / 4)
			y = int(p[i+1, 1] / 4)
			d = dep + rels[0,i,y,x]
			buff[i+1,2] = d 
		buff[:,:2] = p[:,:2]
		results.append(buff)
	return results

def normalize_pts(pose3d, pelvis):
	# pose: [bsize, 17, 3]
	pose3d = pose3d - pose3d[pelvis:pelvis+1,:]
	pose = pose3d[:,:2]
	depth = pose3d[:,2:3]
	xs = pose[:,0:1]
	ys = pose[:,1:2]
	ls = np.sqrt(xs[1:]**2 + ys[1:]**2)
	scale = np.sum(ls, axis=0, keepdims=True)
	scale = scale / (len(ls) - 1)
	pose = pose/scale
	result = np.concatenate([pose, depth], axis=1)
	return result
