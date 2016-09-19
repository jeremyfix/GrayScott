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
else:
    d = 1.5
    N = 256
    dt = 10
pattern = 'solitons'

if(mode <= 2):
    model = grayscott.Model(pattern, N=N, mode=mode, d=d, dt=dt)
elif mode == 3:
    model = libgrayscott.GrayScott(pattern, N, d, dt)
else:
    model = grayscott.SpectralModel(pattern, N=N, d=d, dt=dt, mode='ETDFD')

model.init()

def make_effect2(u_orig):
    #u = cv2.resize(u_orig, (300,300))
    u = u_orig.copy()
    u[u >= 0.5] = 1.0
    u[u < 0.5] = 0
    
    kernel = np.zeros((10,10), dtype=np.float)
    kernel[:8,:] = -1
    kernel[8:, :] = 1
    effect = scipy.signal.convolve2d(2. * (u - 0.5), kernel, mode='same')
    effect /= effect.max()
    effect[effect <= 0.0] = 0.0
    dst = 0.3 * u + 0.7 * effect
    # Edge enhancement
    #dst = cv2.resize(dst, (500, 500))
    #kernel = 0.25 * np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float)
    #dst = dst + scipy.signal.convolve2d(dst, kernel, mode='same')
    #print(dst.min(), dst.max())
    #dst[dst <= 0] = 0
    #dst += dst.min()
    #dst /= dst.max()
    dst[dst >= 1.0] = 1.0
    
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


u = np.zeros((N, N))
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
    u_img = make_effect2(u)
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
