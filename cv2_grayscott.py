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
print("   c : erase the reactant v in a randomly chosen circular patch")
    
cv2.namedWindow('u')
#cv2.namedWindow('v')

key = 0
run = False

mode = int(sys.argv[1])
#
d = 1.0 # The width of the domain
N = 100 # The size of the lattice
dt = 1.0 # the time step
pattern = 'worms'

if(mode <= 2):
    model = grayscott.Model(pattern, N=N, mode=mode, d=d, dt=dt)
else:
    model = libgrayscott.GrayScott(pattern, N, d, dt)

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

