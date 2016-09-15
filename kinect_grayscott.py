# coding: utf-8
# Simulation of the gray scott reaction diffusion system.
#      /  
#      |  ∂ₜu(x,t) = Dᵤ ∇²u(x,t) - u(x,t) v²(x,t) + F(1- u(x,t))
#      |  ∂ₜv(x,t) = Dᵥ ∇²v(x,t) + u(x,t) v²(x,t) - (F + k) v(x,t)
#      \


import cv2
import sys
import random
import numpy as np
import time
from freenect import sync_get_depth as get_depth

import grayscott

print(" Press : ")
print("   s : start/pause")
print("   i : reinitialize the concentrations")
print("   q : quit")
print("   f : toggle fullscreen/normal screen")

cv2.namedWindow("Depth")
cv2.namedWindow('u', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("u", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

key = 0
run = False

#
d = 1.5
N = 256
dt = 10
pattern = 'solitons'

model = grayscott.SpectralModel(pattern, N=N, d=d, dt=dt, mode='ETDFD')

model.init()


u = np.zeros((N, N))
epoch = 0
can_mask = False

while key != ord('q'):

    # As soon as we get a minimum reactant, we start
    # to take into account the kinect
    # Otherwise, the whole activity vanishes
    if((model.get_ut().mean() <= 0.7) and not can_mask):
        can_mask = True
        print("Masking begins")
    if(run):
        if(can_mask):
            (depth,_) = get_depth()
            
            depth_img = (np.dstack((depth, depth, depth)).astype(np.float)/2048.)
            cv2.resize(depth_img, (N, N))
            cv2.imshow('Depth', depth_img)
            
            depth = depth.astype(np.float)/2048.
	    depth = 1. - cv2.resize(depth, (N, N))
            #print(depth.min(), depth.max(), depth.mean())
            depth = depth * 0.85 / depth.mean()
            mask = 0.75 + 0.25 * depth
            model.mask_reactant(mask)
        model.step()
        u[:,:] = model.get_ut()
	epoch += 1

    cv2.imshow('u', u)

    key = cv2.waitKey(1) & 0xFF

    if(key == ord('c')):
        c = (random.randint(0, N-1), random.randint(0, N-1))
        model.erase_reactant(c , N/8)
    elif(key == ord('m')):
        mask = 0.75 + 0.25*np.random.random((N, N))
        model.mask_reactant(mask)
    elif key == ord('s'):
        run = not run
        print("Running ? : " + str(run))
    elif key == ord('i'):
        model.init()
    elif key == ord('f'):
        screenmode = cv2.getWindowProperty("u", cv2.WND_PROP_FULLSCREEN)
        if(screenmode == cv2.WINDOW_NORMAL):
            cv2.setWindowProperty("u", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty("u", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
