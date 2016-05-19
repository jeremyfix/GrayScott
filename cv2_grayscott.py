import cv2
import sys
import random
import numpy as np

import grayscott
import libgrayscott

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

while key != ord('q'):
    if(run): 
        if(mode <= 2):
            model.step()	
            u[:,:] = model.ut
        else:
            u = model.step()

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

