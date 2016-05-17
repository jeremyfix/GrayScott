
import numpy as np
import scipy.signal
import scipy.ndimage
import cv2
import random
import numpy as np

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
#F = 0.0580
#k = 0.0630


# Spirals and wave fronts
# for the initialization, you should not use the square
# but perturb it with some circles (press 'c' while the app is running)
F = 0.0060
k = 0.0370

F = 0.0620
k  = 0.0609


noise = 0.1

dN = N/4

stencil = np.zeros((N, N))
stencil[0,0] = -4
stencil[0,1] = 1
stencil[0,-1] = 1
stencil[1,0] = 1
stencil[-1,0] = 1
fft_mask = np.fft.rfft2(stencil)

#def laplacian(x):
#	return np.fft.irfft2(np.fft.rfft2(x)*fft_mask)

#def laplacian(x):
#	stencil = np.array([[0, 1., 0], [1., -4., 1.], [0, 1., 0]], dtype=float)
#	return scipy.ndimage.convolve(x, stencil, mode='wrap')

def laplacian(x):
	return scipy.ndimage.laplace(x, mode='wrap')

def step(u, v, Du, Dv, F, k, dt):
	return u + dt * (Du * laplacian(u) - u * v**2 + F * (1 - u)), \
		   v + dt * (Dv * laplacian(v) + u * v**2 - (F + k) * v)

def init(noise=0.01):
	u = np.ones((N, N))
	u[(N/2 - dN/2): (N/2+dN/2+1), (N/2 - dN/2) : (N/2+dN/2+1)] = 0.5
	u += noise * (2 * np.random.random((N, N)) - 1)
	u[u <= 0] = 0

	v = np.zeros((N, N))
	v[(N/2 - dN/2): (N/2+dN/2+1), (N/2 - dN/2) : (N/2+dN/2+1)] = 0.25
	v += noise * (2 * np.random.random((N, N)) - 1)
	v[v <= 0] = 0

	return u, v


cv2.namedWindow('u')
cv2.namedWindow('v')

key = 0
run = False
epoch = 0

u, v = init()

while key != ord('q'):
	if(run): 
		print(epoch)
		epoch += 1
		u, v = step(u, v, Du, Dv, F, k, dt)
	cv2.imshow('u', u)
	cv2.imshow('v', v)
	key = cv2.waitKey(1) & 0xFF
	if(key == ord('c')):
		c = (random.randint(0, N-1), random.randint(0, N-1))
		cv2.circle(u, c , dN, 0, -1)
	elif key == ord('s'):
		run = not run
		print("Running ? : " + str(run))
	elif key == ord('i'):
		u, v = init()

