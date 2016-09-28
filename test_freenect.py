import freenect
from freenect import sync_get_depth as get_depth
import freenect
import cv2
import numpy as np

#mdev = freenect.open_device(freenect.init(), 0)
#freenect.set_depth_mode(mdev, freenect.RESOLUTION_MEDIUM, freenect.DEPTH_REGISTERED)

def get_depth_meters():
    depth, _ = get_depth(format=freenect.DEPTH_MM)
    return depth*1e-3

cv2.namedWindow("Depth")
zmin = 1
zmax = 2
key = 0
while(key != ord('q')):
	#(depth,_) = get_depth()
        #depth /= 2048.

        depth = get_depth_meters()
        # Restrict to box in [zmin; zmax]
        depth = (zmax - depth)/(zmax - zmin)
        depth[depth < 0] = 0
        depth[depth > 1] = 0


	print(depth.min(), depth.max())
	# Build up a RGB image with components in [0,1]
	depth_img = (np.dstack((depth, depth, depth)).astype(np.float))
	depth_img = cv2.resize(depth_img, (500, 500))
        #print(depth_img)
	cv2.imshow('Depth', depth_img)
	key = cv2.waitKey(1) & 0xFF
