
import numpy as np
import scipy.signal
import scipy.ndimage
import cv2
import random
import numpy as np
import time
import sys

N = 128
h = 0.01

Du = 2*1e-5 / h**2 
Dv = 1e-5 / h**2

dt = 1.0

F = 0.040
k = 0.060

# Solitons
# k = 0.056
# F = 0.020

# worms
F = 0.0580
k = 0.0630


# Spirals and wave fronts
# for the initialization, you should not use the square
# but perturb it with some circles (press 'c' while the app is running)
#F = 0.0060
#k = 0.0370

#F = 0.0620
#k  = 0.0609


noise = 0.1

dN = N/4

stencil = np.zeros((N, N))
stencil[0,0] = -4
stencil[0,1] = 1
stencil[0,-1] = 1
stencil[1,0] = 1
stencil[-1,0] = 1
fft_mask = np.fft.rfft2(stencil)

mode = int(sys.argv[1])

# On 128 x 128
# mode 1 : 300 fps

if(mode == 0):	
	def laplacian(x):
		return np.fft.irfft2(np.fft.rfft2(x)*fft_mask)
elif mode == 1:
	def laplacian(x):
		stencil = np.array([[0, 1., 0], [1., -4., 1.], [0, 1., 0]], dtype=float)
		return scipy.ndimage.convolve(x, stencil, mode='wrap')
elif mode == 2:
	def laplacian(x):
		return scipy.ndimage.laplace(x, mode='wrap')

def step(ut_1, vt_1, ut, vt, Du, Dv, F, k, dt):
	uvv = ut_1 * vt_1**2
	lu = laplacian(ut_1)
	lv = laplacian(vt_1)	
	ut[:,:] = ut_1 + dt * (Du * lu - uvv + F*(1-ut_1))
	vt[:,:] = vt_1 + dt * (Dv * lv + uvv - (F + k) * vt_1)

def init(u, v, noise=0.01):
	u[:,:] = 1
	u[(N/2 - dN/2): (N/2+dN/2+1), (N/2 - dN/2) : (N/2+dN/2+1)] = 0.5
	u += noise * (2 * np.random.random((N, N)) - 1)
	u[u <= 0] = 0

	v[:,:] = 0
	v[(N/2 - dN/2): (N/2+dN/2+1), (N/2 - dN/2) : (N/2+dN/2+1)] = 0.25
	v += noise * (2 * np.random.random((N, N)) - 1)
	v[v <= 0] = 0

	return u, v


cv2.namedWindow('u')
cv2.namedWindow('v')

key = 0
run = False
epoch = 0

ut_1 = np.zeros((N, N), dtype=float)
vt_1 = np.zeros((N, N), dtype=float)
ut = np.zeros((N, N), dtype=float)
vt = np.zeros((N, N), dtype=float)
init(ut_1, vt_1)

t0 = time.time()

while key != ord('q'):
	if(run): 
		epoch += 1
		step(ut_1, vt_1, ut, vt, Du, Dv, F, k, dt)
		ut_1, vt_1 = ut, vt
		if(epoch % 100 == 0):
			t1 = time.time()
			print("FPS : %i fps" % (100 / (t1 - t0)))
			t0 = t1
	cv2.imshow('u', ut)
	cv2.imshow('v', vt)
	key = cv2.waitKey(1) & 0xFF

	if(key == ord('c')):
		c = (random.randint(0, N-1), random.randint(0, N-1))
		cv2.circle(ut_1, c , dN, 0, -1)
	elif key == ord('s'):
		run = not run
		print("Running ? : " + str(run))
	elif key == ord('i'):
		init(ut_1, vt_1)

