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

if(len(sys.argv) <= 1):
    print("Usage : %s pattern "% sys.argv[0])
    sys.exit(-1)

ipattern = int(sys.argv[1])
if(ipattern == 0):
    pattern = 'solitons'
elif(ipattern == 1):
    pattern = 'worms'
else:
    pattern = 'spirals'
print(" ************** ")
print("       {:^}      ".format(pattern))
print(" ************** ")

    
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

d = 1.
height = 128
width = 256
dt = 10
display_scaling_factor = 4

model = grayscott.SpectralModel(pattern, width=width, height=height, d=d, dt=dt, mode='ETDFD')

model.init()

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
    dst[dst >= 1] = 1
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
        u = model.get_vt()
        u_img = cv2.resize(1-u, (display_scaling_factor*u.shape[1], display_scaling_factor*u.shape[0]),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite("u-%05d.png" % frame_id, (255*u_img).astype(np.uint8))
        
        v = model.get_vt()
        v_img = cv2.resize(1.-v, (display_scaling_factor*v.shape[1], display_scaling_factor*v.shape[0]),interpolation=cv2.INTER_CUBIC)
        print("Saving v-%05d.png" % frame_id)
        cv2.imwrite("v-%05d.png" % frame_id, (255*v_img).astype(np.uint8))
        frame_id += 1
    elif key == ord('f'):
        screenmode = cv2.getWindowProperty("u", cv2.WND_PROP_FULLSCREEN)
        if(screenmode == normal_flag):
            cv2.setWindowProperty("u", cv2.WND_PROP_FULLSCREEN, fullscreen_flag)
        else:
            cv2.setWindowProperty("u", cv2.WND_PROP_FULLSCREEN, normal_flag)
