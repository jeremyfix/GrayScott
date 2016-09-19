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

#img[img > 0.5] = 1
#img[img <= 0.5] = 0
img = 2. * (img - 0.5)  # img in [-1,1]

kernel = np.zeros((10, 10), dtype=np.float)
kernel[:8,:] = -1
kernel[8:,:] = 1

effect = scipy.signal.convolve2d(img, kernel, mode='same')
#dst[dst < 0] = 0
effect /= effect.max()
effect[effect <= 0.0] = 0
effect_hires = cv2.resize(effect, (700, 700), interpolation=cv2.INTER_CUBIC)


img_hires = cv2.resize(img, (700, 700),interpolation=cv2.INTER_CUBIC)
img_hires[img_hires > 0.] = 1.
img_hires[img_hires <= 0.] = 0.

img_hires_blur = scipy.signal.convolve2d(img_hires, np.ones((11,11))/121., mode='same')


dst = 0.6 * img_hires + 0.4 * effect_hires
dst[img_hires >= 0.99] = img_hires_blur[img_hires >= 0.99]


cv2.imshow("src", img_hires)
cv2.imshow("effect", effect_hires)
cv2.imshow("dst", dst)
cv2.waitKey(0)


