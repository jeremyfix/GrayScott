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
import scipy

import grayscott
import libgrayscott

if(len(sys.argv) <= 1):
    print("Usage : %s mode "% sys.argv[0])
    print("With mode : ")
    print("   0 : spatial model with FFT convolution in python, forward euler") # 100 fps
    print("   1 : spatial model with ndimage.convolve in python, forward euler") # 165 fps
    print("   2 : spatial model with ndimage.laplace in python, forward euler") # 150 fps
    print("   3 : spatial model with fast laplacian in C++, forward euler") # 400 fps
    print("   4 : spectral model in python using ETDRK4")
    sys.exit(-1)

print(" Press : ")
print("   s : start/pause")
print("   i : reinitialize the concentrations")
print("   q : quit")
print("   c : erase the reactant v in a randomly chosen box patch")
print("   m : mask the reactant with a randomly generated mask")
print("   p : save the current u potential")
print("   f : toggle fullscreen/normal screen")
    

try:
    fullscreen_flag = cv2.WINDOW_FULLSCREEN
    normal_flag = cv2.WINDOW_NORMAL
except:
    fullscreen_flag = cv2.cv.CV_WINDOW_FULLSCREEN
    normal_flag = cv2.cv.CV_WINDOW_NORMAL

cv2.namedWindow('u', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("u", cv2.WND_PROP_FULLSCREEN, normal_flag)

key = 0
run = False

mode = int(sys.argv[1])
#
if(mode <= 3):
    d = 1.5 # The width of the domain
    height = 128 # The size of the lattice
    width = 256
    dt = 0.1 # the time step
else:
    d = 3.
    height = 128
    width = 256
    dt = 10
display_scaling_factor = 4
pattern = 'solitons'

if(mode <= 2):
    model = grayscott.Model(pattern, width=width, height=height, mode=mode, d=d, dt=dt)
elif mode == 3:
    model = libgrayscott.GrayScott(pattern, width, height, d, dt)
else:
    model = grayscott.SpectralModel(pattern, width=width, height=height, d=d, dt=dt, mode='ETDFD')

model.init()

# Precompute the FFT of the kernel for speeding up the convolution
# For lightning
s_kernel = 11
kernel_light = np.ones((s_kernel,s_kernel), dtype=np.float)
kernel_light[:int(2./3 * s_kernel),:int(2./3 * s_kernel)] = -1
mask_light = np.zeros((height, width), np.float32)
mask_light[0:kernel_light.shape[0], 0:kernel_light.shape[1]] = kernel_light
mask_light = np.roll(np.roll(mask_light,-(kernel_light.shape[1]//2-1), axis=1),-(kernel_light.shape[0]//2-1), axis=0)
fft_mask_light = np.fft.rfft2(mask_light)
# For the blur, it is fastest to use uniform_filter


def make_effect(u_orig, scale):
    res_height, res_width = scale * u_orig.shape[0], scale * u_orig.shape[1]

    # Compute the lightning effect
    effect = np.fft.irfft2(np.fft.rfft2(2.*(u_orig-0.5))* fft_mask_light)
    
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
    dst[dst > 1] = 1
    dst[dst < 0] = 0
    return dst


u = np.zeros((height, width))
epoch = 0

t0 = time.time()
frame_id = 0
while key != ord('q'):
    if(run): 
        model.step()
        u[:,:] = model.get_ut()
	epoch += 1
	if(epoch % 100 == 0):
		t1 = time.time()
		print("FPS: %f fps" % (100 / (t1 - t0)))
		t0 = t1
        
        u_img = make_effect(u, display_scaling_factor)
        cv2.imshow('u', u_img)

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
    elif key == ord('p'):
        print("Saving u-%05d.png" % frame_id)
        cv2.imwrite("u-%05d.png" % frame_id, (np.minimum(255*u_img, 255)).astype(np.uint8))
        frame_id += 1
    elif key == ord('f'):
        screenmode = cv2.getWindowProperty("u", cv2.WND_PROP_FULLSCREEN)
        if(screenmode == normal_flag):
            cv2.setWindowProperty("u", cv2.WND_PROP_FULLSCREEN, fullscreen_flag)
        else:
            cv2.setWindowProperty("u", cv2.WND_PROP_FULLSCREEN, normal_flag)
