
import numpy as np
import scipy.signal
import scipy.ndimage
import numpy as np
import time
import sys

class Model:

	def __init__(self, param_name, N, mode, measure_fps):
		self.param_name = param_name
		if(self.param_name == 'solitons'):
			self.k = 0.056
			self.F = 0.020
 		elif(self.param_name == 'worms'):
			self.k = 0.0630
			self.F = 0.0580
		elif(self.param_name == 'spirals'):			
			self.k = 0.0370
			self.F = 0.0060
		else:
			self.k = 0.040
			self.F = 0.060
		self.N = N
		self.h = 1e-2		
		self.Du = 2 * 1e-5 / self.h**2
		self.Dv = 1e-5 / self.h**2
		self.dt = 1.0
		self.noise = 0.1

		self.ut_1 = np.zeros((self.N, self.N), dtype=float)
		self.vt_1 = np.zeros((self.N, self.N), dtype=float)
		self.ut = np.zeros((self.N, self.N), dtype=float)
		self.vt = np.zeros((self.N, self.N), dtype=float)

		self.mode = mode
		if(self.mode == 0):
			self.stencil = np.zeros((self.N, self.N))
			self.stencil[0,0] = -4
			self.stencil[0,1] = 1
			self.stencil[0,-1] = 1
			self.stencil[1,0] = 1
			self.stencil[-1,0] = 1
			self.fft_mask = np.fft.rfft2(self.stencil)
		elif(self.mode == 1):
			self.stencil = np.array([[0, 1., 0], [1., -4., 1.], [0, 1., 0]], dtype=float)

		self.measure_fps = measure_fps
		if(measure_fps):
			self.t0 = time.time()
		self.epoch = 0

	def init(self):
		if(self.param_name == 'spirals'):
			self.ut_1[:,:] = np.random.random((self.N, self.N))
			self.vt_1[:,:] = np.random.random((self.N, self.N))
		else:
			dN = self.N/4
			self.ut_1[:,:] = 1
			self.ut_1[(self.N/2 - dN/2): (self.N/2+dN/2+1), (self.N/2 - dN/2) : (self.N/2+dN/2+1)] = 0.5
			self.ut_1 += self.noise * (2 * np.random.random((self.N, self.N)) - 1)
			self.ut_1[self.ut_1 <= 0] = 0

			self.vt_1[:,:] = 0
			self.vt_1[(self.N/2 - dN/2): (self.N/2+dN/2+1), (self.N/2 - dN/2) : (self.N/2+dN/2+1)] = 0.25
			self.vt_1 += self.noise * (2 * np.random.random((self.N, self.N)) - 1)
			self.vt_1[self.vt_1 <= 0] = 0

	def laplacian(self, x):
		if(self.mode == 0):
			return np.fft.irfft2(np.fft.rfft2(x)*self.fft_mask)
		elif(self.mode == 1):
			return scipy.ndimage.convolve(x, self.stencil, mode='wrap')
		elif(self.mode == 2):
			return scipy.ndimage.laplace(x, mode='wrap')



	def step(self):
		uvv = self.ut_1 * self.vt_1**2
		lu = self.laplacian(self.ut_1)
		lv = self.laplacian(self.vt_1)	
		self.ut[:,:] = self.ut_1 + self.dt * (self.Du * lu - uvv + self.F*(1-self.ut_1))
		self.vt[:,:] = self.vt_1 + self.dt * (self.Dv * lv + uvv - (self.F + self.k) * self.vt_1)

		self.epoch += 1
		self.ut_1, self.vt_1  = self.ut, self.vt

		if(self.measure_fps and (self.epoch % 100 == 0)):
			self.t1 = time.time()
			print("FPS : %f f/s" % (100 / (self.t1 - self.t0)))
			self.t0 = self.t1



if(__name__ == '__main__'):
	mode = int(sys.argv[1])

	model = Model('worms', N=128, mode=mode, measure_fps=True)
	model.init()

	while True:
		model.step()

