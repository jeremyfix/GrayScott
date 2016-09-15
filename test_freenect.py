from freenect import sync_get_depth as get_depth, sync_get_video as get_video
import cv2
import numpy as np



cv2.namedWindow("Depth")

key = 0
while(key != ord('q')):
	(depth,_) = get_depth()
	print(depth.max())
	# Build up a RGB image with components in [0,1]
	depth_img = (np.dstack((depth, depth, depth)).astype(np.float)/2048.)
	depth_img = cv2.resize(depth_img, (100, 100))
        print(depth_img)
	cv2.imshow('Depth', depth_img)
	key = cv2.waitKey(1) & 0xFF
