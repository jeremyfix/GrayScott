import cv2
import sys
import random
import numpy as np
import time

import grayscott
import libgrayscott

if(len(sys.argv) <= 1):
    print("Usage : %s mode "% sys.argv[0])
    print("With mode : ")
    print("   0 : spatial model with FFT convolution in python") # 100 fps
    print("   1 : spatial model with ndimage.convolve in python") # 165 fps
    print("   2 : spatial model with ndimage.laplace in python") # 150 fps
    print("   3 : spatial model with fast laplacian in C++") # 400 fps
    print("   4 : spectral model in python")
    sys.exit(-1)

cv2.namedWindow('u')
#cv2.namedWindow('v')

key = 0
run = False

mode = int(sys.argv[1])
N = 256
pattern = 'worms'

if(mode <= 2):
    model = grayscott.Model(pattern, N=N, mode=mode)
else:
    model = libgrayscott.GrayScott(pattern, N)

model.init()


u = np.zeros((N, N))
epoch = 0

t0 = time.time()
while key != ord('q'):
    if(run): 
        model.step()
        u[:,:] = model.get_ut()
	epoch += 1
	if(epoch % 200 == 0):
		t1 = time.time()
		print("FPS: %f fps" % (200 / (t1 - t0)))
		t0 = t1
    cv2.imshow('u', u)
    #cv2.imshow('v', model.vt)

    key = cv2.waitKey(1) & 0xFF

    if(key == ord('c')):
        c = (random.randint(0, N-1), random.randint(0, N-1))
        cv2.circle(model.vt_1, c , N/4, 0, -1)
    elif key == ord('s'):
        run = not run
        print("Running ? : " + str(run))
    elif key == ord('i'):
        model.init()

