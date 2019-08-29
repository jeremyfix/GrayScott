# coding: utf-8
# Simulation of the gray scott reaction diffusion system.
#      /
#      |  ∂ₜu(x,t) = Dᵤ ∇²u(x,t) - u(x,t) v²(x,t) + F(1- u(x,t))
#      |  ∂ₜv(x,t) = Dᵥ ∇²v(x,t) + u(x,t) v²(x,t) - (F + k) v(x,t)
#      \


import cv2
import numpy as np
import time
import freenect
from freenect import sync_get_depth as get_depth
import scipy
import sys

import grayscott


def get_depth_meters():
    depth, _ = get_depth(format=freenect.DEPTH_MM)
    return depth*1e-3


print("Try to get a depth image from the kinect to"
      "to ensure it works correctly")
try:
    get_depth_meters()
except:
    sys.exit(-1)
print("Ok; I'm happy")

print(" Press : ")
print("   s : start/pause")
print("   i : reinitialize the concentrations")
print("   q : quit")
print("   f : toggle fullscreen/normal screen")

try:
    fullscreen_flag = cv2.WINDOW_FULLSCREEN
    normal_flag = cv2.WINDOW_NORMAL
except:
    fullscreen_flag = cv2.cv.CV_WINDOW_FULLSCREEN
    normal_flag = cv2.cv.CV_WINDOW_NORMAL

# cv2.namedWindow("Depth")
cv2.namedWindow('u', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("u", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

key = 0
run = False

#
d = 3.0
width = 200
height = 100
dt = 10
pattern = 'worms_solitons'
display_scaling_factor = 4
# The frustum for the kinect depth
zmin = 1
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


def insert_text(img, text):
    global height
    img[(img.shape[0]-40):,:] = 1
    cv2.putText(img, text, (20, img.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, thickness=2)

def insert_depth(depth_img, img):
    tgt_size = (2*64, 2*48)
    small_depth = cv2.resize(depth_img, tgt_size)
    #print(small_depth.shape)
    img[(img.shape[0]-small_depth.shape[0]):, (img.shape[1]-small_depth.shape[1]):] = small_depth[:,:, 0]
    return


model = grayscott.SpectralModel(pattern,
                                width, height,
                                d=d, dt=dt, mode='ETDFD')
model = grayscott.ThreadedModel(model)
model.init()
model.start()

u = np.zeros((height, width))

depth = np.ones((height, width))
depth_img = np.zeros((2, 2, 3), dtype=np.float)
can_mask = False

t0 = time.time()
epoch = 0
while key != ord('q'):

    # As soon as we get a minimum reactant, we start
    # to take into account the kinect
    # Otherwise, the whole activity vanishes
    if((model.get_ut().mean() <= 0.9) and not can_mask):
        can_mask = True
        print("Masking begins")
    if not model.is_paused() and can_mask:
        depth = get_depth_meters()
        # Restrict to box in [zmin; zmax]
        # depth is scaled in meters, and the horizontal axis is flipped
        # what is in [zmin, zmax] is rescaled to [1, 0] , the rest set to 0
        depth = (zmax - depth[:, ::-1])/(zmax - zmin)
        depth[depth < 0] = 0
        depth[depth > 1] = 0

        depth_img = (np.dstack((depth, depth, depth)).astype(np.float))
        # cv2.resize(depth_img, (width, height))
        # cv2.imshow('Depth', depth_img)

        depth = cv2.resize(depth.astype(np.float), (width, height))
        # depth = 1. - cv2.resize(depth, (N, N))
        # print(depth.min(), depth.max(), depth.mean())
        # depth = depth * 0.85 / depth.mean()
        # mask = 0.75 + 0.25 * depth
        model.mask_reactant(depth)

    u_img = make_effect(model.get_ut(), display_scaling_factor)
    insert_text(u_img, "GrayScott Reaction Diffusion")
    insert_depth(depth_img, u_img)

    cv2.imshow('u', u_img)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        model.trigger_pause()
        print("Running ? : " + str(not model.is_paused()))
    elif key == ord('i'):
        model.init()
    elif key == ord('f'):
        screenmode = cv2.getWindowProperty("u", cv2.WND_PROP_FULLSCREEN)
        if(screenmode == normal_flag):
            cv2.setWindowProperty("u", cv2.WND_PROP_FULLSCREEN, fullscreen_flag)
        else:
            cv2.setWindowProperty("u", cv2.WND_PROP_FULLSCREEN, normal_flag)

model.stop()
model.join()
