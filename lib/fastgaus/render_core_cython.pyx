import numpy as np 
cimport numpy as np 
import cv2 

cdef extern from "render_core.h":
	void _get_gaus(float* map, int sizeh, int sizew, float sigma, float x, float y)

def render_heatmap(np.ndarray[float, ndim=2, mode='c'] pts, int sizeh, int sizew, float sigma):
	cdef int n_pts = pts.shape[0]
	hmap = np.zeros([n_pts, sizeh, sizew], dtype=np.float32)
	for i in range(n_pts):
		if pts[i,2] > 0:
			_get_gaus(<float*> np.PyArray_DATA(hmap[i]), sizeh, sizew, sigma, pts[i,0], pts[i,1])
	return hmap

def render_paf(np.ndarray[float, ndim=3, mode='c'] pairs, np.ndarray[float, ndim=1, mode='c'] conf, int size, int width):
	n_pairs = pairs.shape[0]
	hmap = np.zeros([n_pairs, 2, size, size], dtype=np.float32)
	buff = np.zeros([size, size], dtype=np.uint8)
	for i in range(n_pairs):
		if conf[i]<=0:
			continue
		p = pairs[i]

		diff = p[0] - p[1]
		height = np.sqrt(np.sum(np.square(diff))) / 2.
		c = 0.5 * (p[0] + p[1])

		l = np.sqrt(np.sum(np.square(diff)))
		if l==0.0:
			continue
		cos = diff[0] / l 
		sin = diff[1] / l 

		pts = np.float32([[-width, height], [width, height], [width, -height], [-width, -height]])
		rot = np.float32([[-sin, cos],[cos, sin]])
		pts = pts.dot(rot.T)
		pts += c 

		pts = pts.astype(int)
		cv2.fillConvexPoly(buff, points=pts, color=1)

		diff = diff / np.linalg.norm(diff)
		hmap[i,0] = np.float32(buff) * diff[0]
		hmap[i,1] = np.float32(buff) * diff[1]
	return hmap