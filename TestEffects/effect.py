import cv2
import sys
import numpy as np
import scipy.signal

if(len(sys.argv) != 2):
	print("Usage : %s img" % sys.argv[0])
	sys.exit(-1)


cv2.namedWindow("src")
cv2.namedWindow("effect")
cv2.namedWindow("dst")

img = (cv2.imread(sys.argv[1])[:,:,0]).astype(np.float)/255.
img = 2. * (img - 0.5)  # img in [-1,1]
#img[img > 0.5] = 1
#img[img <= 0.5] = 0
print(img[0,:])
print(img.shape)


kernel = np.zeros((5, 5), dtype=np.float)
kernel[:3,:] = -1
kernel[3:,:] = 1
print(kernel.mean())

effect = scipy.signal.convolve2d(img, kernel, mode='same')
#dst[dst < 0] = 0
effect /= effect.max()
effect[effect <= 0.0] = 0

img = (img + 1.)/2.

dst = 0.8 * img + 0.2 * effect



cv2.imshow("src", cv2.resize(img, (700, 700), interpolation=cv2.INTER_CUBIC))
cv2.imshow("effect", effect)
cv2.imshow("dst", cv2.resize(dst, (700, 700), interpolation=cv2.INTER_CUBIC))
cv2.waitKey(0)


