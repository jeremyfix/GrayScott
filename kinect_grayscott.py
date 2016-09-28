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
import freenect
from freenect import sync_get_depth as get_depth
import scipy 

import grayscott

print(" Press : ")
print("   s : start/pause")
print("   i : reinitialize the concentrations")
print("   q : quit")
print("   f : toggle fullscreen/normal screen")

try:
    fullscreen_flag = cv2.WINDOW_FULLSCREEN
    normal_flag = CV2.WINDOW_NORMAL
except:
    fullscreen_flag = cv2.cv.CV_WINDOW_FULLSCREEN
    normal_flag = cv2.cv.CV_WINDOW_NORMAL

cv2.namedWindow("Depth")
cv2.namedWindow('u', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("u", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

key = 0
run = False

#
d = 3.0
width = 256
height = 128
dt = 10
pattern = 'spirals'
display_scaling_factor = 4
# The frustum for the kinect depth
zmin = 2
zmax = 4



def make_effect(u_orig, scale):
    res_height, res_width = scale * u_orig.shape[0], scale * u_orig.shape[1]
    s_kernel = 11
    kernel = np.ones((s_kernel,s_kernel), dtype=np.float)
    # Light coming from top left
    kernel[:int(2./3 * s_kernel),:int(2./3 * s_kernel)] = -1
    # Light coming from left
    #kernel[:int(2./3 * s_kernel),:] = -1
    effect = scipy.signal.convolve2d(2. * (u_orig - 0.5), kernel, mode='same')
    effect /= 30. # HAND TUNED SCALING of the effect ... might need to be adapted if changing s_kernel
    effect[effect >= 1.0] = 1.0
    effect[effect <= 0.0] = 0.0
    effect_hires = cv2.resize(effect, (res_width, res_height), interpolation=cv2.INTER_CUBIC)

    u_hires = cv2.resize(u_orig, (res_width, res_height),interpolation=cv2.INTER_CUBIC)
    u_hires[u_hires >= 0.5] = 1.
    u_hires[u_hires < 0.5 ] = 0.
    # Blur the image to get the shading
    u_blur = scipy.ndimage.filters.uniform_filter(u_hires, size=5)
    # Shift the shadding down right
    u_blur = np.lib.pad(u_blur, ((2,0),(2,0)), 'constant', constant_values=1)[:-2,:-2]
    
    dst = 0.6 * u_hires + 0.4 * effect_hires
    dst[u_hires >= 0.99] = u_blur[u_hires >= 0.99]
    return dst




model = grayscott.SpectralModel(pattern, width, height, d=d, dt=dt, mode='ETDFD')
model.init()


u = np.zeros((height, width))
epoch = 0
can_mask = False

while key != ord('q'):

    # As soon as we get a minimum reactant, we start
    # to take into account the kinect
    # Otherwise, the whole activity vanishes
    if((model.get_ut().mean() <= 0.9) and not can_mask):
        can_mask = True
        print("Masking begins")
    if(run):
        if(can_mask):
            (depth,_) = get_depth(format=freenect.DEPTH_MM)
            # Restrict to box in [zmin; zmax]
            # depth is scaled in meters, and the horizontal axis is flipped
            # what is in [zmin, zmax] is rescaled to [1, 0] , the rest set to 0
            depth = (zmax - depth[:,::-1]*1e-3)/(zmax - zmin)
            depth[depth < 0] = 0
            depth[depth > 1] = 0

            depth_img = (np.dstack((depth, depth, depth)).astype(np.float))
            cv2.resize(depth_img, (width, height))
            cv2.imshow('Depth', depth_img)
            
            depth = cv2.resize(depth.astype(np.float), (width, height))
	    #depth = 1. - cv2.resize(depth, (N, N))
            #print(depth.min(), depth.max(), depth.mean())
            #depth = depth * 0.85 / depth.mean()
            #mask = 0.75 + 0.25 * depth
            model.mask_reactant(depth)
        model.step()
        u[:,:] = model.get_ut()
	epoch += 1

    cv2.imshow('u', make_effect(u, display_scaling_factor))

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
        if(screenmode == normal_flag):
            cv2.setWindowProperty("u", cv2.WND_PROP_FULLSCREEN, fullscreen_flag)
        else:
            cv2.setWindowProperty("u", cv2.WND_PROP_FULLSCREEN, normal_flag)
