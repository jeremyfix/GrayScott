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
    
cv2.namedWindow('u', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("u", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

key = 0
run = False

mode = int(sys.argv[1])
#
if(mode <= 3):
    d = 1.5 # The width of the domain
    N = 128 # The size of the lattice
    dt = 1. # the time step
    raise Exception("Must adapt the scripts to handle width/height rather than N..")
else:
    d = 1.5
    height = 128
    width = 256
    dt = 10
pattern = 'solitons'

if(mode <= 2):
    model = grayscott.Model(pattern, N=N, mode=mode, d=d, dt=dt)
elif mode == 3:
    model = libgrayscott.GrayScott(pattern, N, d, dt)
else:
    model = grayscott.SpectralModel(pattern, width=width, height=height, d=d, dt=dt, mode='ETDFD')

model.init()

def make_effect2(u_orig, w=700):
    #u = cv2.resize(u_orig, (300,300))
    #u = u_orig.copy()
    #u[u >= 0.5] = 1.0
    #u[u < 0.5] = 0
    
    kernel = np.zeros((10,10), dtype=np.float)
    kernel[:8,:] = -1
    kernel[8:, :] = 1
    effect = scipy.signal.convolve2d(2. * (u_orig - 0.5), kernel, mode='same')
    #effect /= effect.max()
    effect /= 30.
    effect[effect >= 1.0] = 1.0
    effect[effect <= 0.0] = 0.0
    dst = 0.6 * cv2.resize(u, (w,w), interpolation=cv2.INTER_CUBIC) + 0.4 * cv2.resize(effect, (w,w), interpolation=cv2.INTER_CUBIC)
    # Edge enhancement
    #dst = cv2.resize(dst, (500, 500))
    #kernel = 0.25 * np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float)
    #dst = dst + scipy.signal.convolve2d(dst, kernel, mode='same')
    #print(dst.min(), dst.max())
    #dst[dst <= 0] = 0
    #dst += dst.min()
    #dst /= dst.max()
    #dst[dst >= 1.0] = 1.0
    #dst[dst < 0.] = 0.
    
    return dst#cv2.resize(dst, (1000,1000), interpolation=cv2.INTER_CUBIC)

def make_effect(u_orig):
    u = u_orig.copy()
    kernel = np.zeros((5,5), dtype=np.float)
    kernel[:3,:] = -1
    kernel[3:, :] = 1
    effect = scipy.signal.convolve2d(2. * (u - 0.5), kernel, mode='same')
    effect /= effect.max()
    effect[effect <= 0.0] = 0.0
    dst = 0.8 * u + 0.2 * effect
    # Edge enhancement
    dst = cv2.resize(dst, (500, 500))
    kernel = 0.25 * np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float)
    dst = dst + scipy.signal.convolve2d(dst, kernel, mode='same')
    print(dst.min(), dst.max())
    dst[dst <= 0] = 0
    #dst += dst.min()
    dst /= dst.max()
        
    return cv2.resize(dst, (1000,1000), interpolation=cv2.INTER_CUBIC)

def make_effect3(u_orig, scale):
    res_height, res_width = scale * u_orig.shape[0], scale * u_orig.shape[1]
    kernel = np.zeros((11,11), dtype=np.float)
    kernel[:8,:] = -1
    kernel[8:, :] = 1
    effect = scipy.signal.convolve2d(2. * (u_orig - 0.5), kernel, mode='same')
    effect /= 30.
    effect[effect >= 1.0] = 1.0
    effect[effect <= 0.0] = 0.0
    effect_hires = cv2.resize(effect, (res_width, res_height), interpolation=cv2.INTER_CUBIC)

    u_hires = cv2.resize(u_orig, (res_width, res_height),interpolation=cv2.INTER_CUBIC)
    u_hires[u_hires >= 0.5] = 1.
    u_hires[u_hires < 0.5 ] = 0.
    # Blur the image to get the shading
    u_blur = scipy.ndimage.filters.uniform_filter(u_hires, size=11)
    # Shift the shadding down right
    u_blur = np.lib.pad(u_blur, ((2,0),(2,0)), 'constant', constant_values=1)[:-2,:-2]
    
    dst = 0.6 * u_hires + 0.4 * effect_hires
    #print(dst.shape)
    #print(u_hires.shape)
    dst[u_hires >= 0.99] = u_blur[u_hires >= 0.99]
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
        
        u_img = make_effect3(u,1)
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
        cv2.imwrite("u-%05d.png" % frame_id, (255*u_img).astype(np.uint8))
        frame_id += 1
    elif key == ord('f'):
        screenmode = cv2.getWindowProperty("u", cv2.WND_PROP_FULLSCREEN)
        if(screenmode == cv2.WINDOW_NORMAL):
            cv2.setWindowProperty("u", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty("u", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
