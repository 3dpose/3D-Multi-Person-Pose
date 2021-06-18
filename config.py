import numpy as np 

# headnet 
head_layernum = 1
head_chn = 32

# upsmaple 
upsample_layers = 1
upsample_chn = 32

# size
inp_size = 512 
out_size = 256
base_sigma = 4.0
num_pts = 14

# extra 
max_inst = 30
depth_scaling = 2500
depth_mean = 1.45
rel_depth_scaling = 1000
tag_thresh = 1.0
tag_distance = 1.5
hmap_thresh = 0.1
nms_kernel = 5 
