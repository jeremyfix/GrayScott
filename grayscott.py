# coding: utf-8

import numpy as np
import scipy.signal
import scipy.ndimage
import numpy as np
import time
import sys

import libgrayscott

''' 
We considered a spatial domain of size d × d, with N × N samples

Expressed in the spectral domain, the system we solve is 

 ∂ₜ U [k₁,k₂] = -[Dᵤ ( (2πk₁/d)^2 + (2πk₂/d)^2)] U[k₁,k₂] - TF[TF^-1(U) (TF^-1(V))^2] + F δ_{k₁,k₂} - F U[k₁,k₂] 
 ∂ₜ V [k₁,k₂] = -[Dᵥ ( (2πk₁/d)^2 + (2πk₂/d)^2)] V[k₁,k₂] + TF[TF^-1(U) (TF^-1(V))^2] - (F + k) V[k₁,k₂]

with U = TF(u) , V = TF(v)

If we decompose the linear and non-linear parts of the equations, following the notations of "Fourth order time-stepping for stiff PDEs, the system reads:

 ∂ₜ U[k₁, k₂] = Lᵤ U[k₁,k₂] + Nᵤ(U[k₁,k₂], V[k₁,k₂]) + F δ_{k₁,k₂}
 ∂ₜ V[k₁, k₂] = Lᵥ V[k₁,k₂] - Nᵤ(U[k₁,k₂], V[k₁,k₂])

with Lᵤ U[k₁,k₂]            = -[Dᵤ ( (2πk₁/d)^2 + (2πk₂/d)^2) + F] U[k₁,k₂]
     Nᵤ(U[k₁,k₂], V[k₁,k₂]) = -TF[TF^-1(U) (TF^-1(V))^2]
     Lᵥ V[k₁,k₂]            = -[Dᵥ ( (2πk₁/d)^2 + (2πk₂/d)^2) + (F + k)] V[k₁,k₂]
     
we can then use the formalas of Cox
'''
class SpectralModel:
    def __init__(self, param_name, N, d=1, dt=0.1):
        self.param_name = param_name
        if(self.param_name == 'solitons'):
            self.k = 0.056
            self.F = 0.020
        elif(self.param_name == 'worms'):
            self.k = 0.0630
            self.F = 0.0580
        elif(self.param_name == 'spirals'):			
            self.k = 0.050
            self.F = 0.018
        else:
            self.k = 0.040
            self.F = 0.060
        self.N = N
        self.Du = 0.2
        self.Dv = 0.1
        self.dt = dt
        self.noise = 0.2

        self.ut_1 = np.zeros((self.N, self.N), dtype=float)
        self.vt_1 = np.zeros((self.N, self.N), dtype=float)
        self.ut = np.zeros((self.N, self.N), dtype=float)
        self.vt = np.zeros((self.N, self.N), dtype=float)

        # Precompute various ETDRK4 scalar quantities
        self.k = np.arange(self.N)

    def init(self):
        dN = self.N/4
        self.ut_1[:,:] = 1
        self.ut_1[(self.N/2 - dN/2): (self.N/2+dN/2+1), (self.N/2 - dN/2) : (self.N/2+dN/2+1)] = 0.5
        self.ut_1 += self.noise * (2 * np.random.random((self.N, self.N)) - 1)
        self.ut_1[self.ut_1 <= 0] = 0
        self.vt_1[:,:] = 0
        self.vt_1[(self.N/2 - dN/2): (self.N/2+dN/2+1), (self.N/2 - dN/2) : (self.N/2+dN/2+1)] = 0.25
        self.vt_1 += self.noise * (2 * np.random.random((self.N, self.N)) - 1)
        self.vt_1[self.vt_1 <= 0] = 0
        
        self.tf_ut_1 = np.fft.fft2(self.ut_1)
        self.tf_vt_1 = np.fft.fft2(self.vt_1)
        
        
    def step(self):
        pass
        
class Model:

	def __init__(self, param_name, N, mode,d=1,dt=0.1):
		self.param_name = param_name
		if(self.param_name == 'solitons'):
			self.k = 0.056
			self.F = 0.020
 		elif(self.param_name == 'worms'):
			self.k = 0.0630
			self.F = 0.0580
		elif(self.param_name == 'spirals'):			
			self.k = 0.0500
			self.F = 0.0180
                elif(self.param_name == 'uskate'):
                    self.k = 0.06093
                    self.F = 0.0620
		else:
			self.k = 0.040
			self.F = 0.060
		self.N = N
		self.h = d/N		
		self.Du = 2 * 1e-5 / self.h**2
		self.Dv = 1e-5 / self.h**2
		self.dt = dt
		self.noise = 0.2
                
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


	def init(self):
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

        def get_ut(self):
                return self.ut

	def step(self):
		uvv = self.ut_1 * self.vt_1**2
		lu = self.laplacian(self.ut_1)
		lv = self.laplacian(self.vt_1)	
		self.ut[:,:] = self.ut_1 + self.dt * (self.Du * lu - uvv + self.F*(1-self.ut_1))
		self.vt[:,:] = self.vt_1 + self.dt * (self.Dv * lv + uvv - (self.F + self.k) * self.vt_1)

		self.ut_1, self.vt_1  = self.ut, self.vt



if(__name__ == '__main__'):

    if(len(sys.argv) <= 1):
        print("Usage : %s mode "% sys.argv[0])
        print("With mode : ")
        print("   0 : spatial model with FFT convolution in python, forward euler") # 100 fps
        print("   1 : spatial model with ndimage.convolve in python, forward euler") # 165 fps
        print("   2 : spatial model with ndimage.laplace in python, forward euler") # 150 fps
        print("   3 : spatial model with fast laplacian in C++, forward euler") # 400 fps
        print("   4 : spectral model in python using ETDRK4")
        sys.exit(-1)

    mode = int(sys.argv[1])
    
    N = 256
    pattern = 'worms'
    
    if(mode <= 2):
        model = Model(pattern, N=N, mode=mode)
    elif mode == 3:
        model = libgrayscott.GrayScott(pattern, N)
    elif mode == 4:
        model = SpectralModel(pattern, N=N)
        
    model.init()
    
    epoch = 0
    t0 = time.time()
    
    while True:
        model.step()
        epoch += 1
        if(epoch % 500 == 0):
            t1 = time.time()
            print("FPS : %f f/s" % (500 / (t1 - t0)))
            t0 = t1



