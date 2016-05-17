
import numpy as np
import scipy.signal
import cv2

N = 256
Du = 2*1e-5  * N**2
Dv = 1e-5  * N**2
F = 0.0620
k = 0.0609
dt = 0.1

dN = 20

u = np.ones((N, N))
u[(N/2 - dN/2): (N/2+dN/2+1), (N/2 - dN/2) : (N/2+dN/2+1)] = 0.5 + 0.01 * (2 * np.random.random((dN+1, dN+1)) - 1)
v = np.zeros((N, N))
v[(N/2 - dN/2): (N/2+dN/2+1), (N/2 - dN/2) : (N/2+dN/2+1)] = 0.25 + 0.01 * (2 * np.random.random((dN+1, dN+1)) - 1)

def laplacian(x):
	return scipy.ndimage.laplace(x, mode='wrap')

def step(u, v, Du, Dv, F, k, dt):
	return u + dt * (Du * laplacian(u) - u * v**2 + F * (1 - u)), \
		   v + dt * (Dv * laplacian(v) + u * v**2 - (F + k) * v)

cv2.namedWindow('u')

key = 0

while key != ord('q'):
	u, v = step(u, v, Du, Dv, F, k, dt)
	print(u.min(), u.max())
	cv2.imshow('u', u)
	key = cv2.waitKey(1) & 0xFF

