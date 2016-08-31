# coding: utf-8

import numpy as np
import scipy.signal
import scipy.ndimage
import numpy as np
import time
import sys

import libgrayscott

''' 
We consider a spatial domain of size d × d, with N × N samples

Expressed in the spectral domain, the system we solve is 

 ∂ₜ U [k₁,k₂] = -[Dᵤ ( (2πk₁/d)^2 + (2πk₂/d)^2)] U[k₁,k₂] - TF[TF^-1(U) (TF^-1(V))^2] + F N^2 δ_{k₁,k₂} - F U[k₁,k₂] 
 ∂ₜ V [k₁,k₂] = -[Dᵥ ( (2πk₁/d)^2 + (2πk₂/d)^2)] V[k₁,k₂] + TF[TF^-1(U) (TF^-1(V))^2] - (F + k) V[k₁,k₂]

with U = TF(u) , V = TF(v)

If we decompose the linear and non-linear parts of the equations, following the notations of "Fourth order time-stepping for stiff PDEs, the system reads:

 ∂ₜ U[k₁, k₂] = Lᵤ U[k₁,k₂] + Nᵤ(U[k₁,k₂], V[k₁,k₂]) + F N^2 δ_{k₁,k₂}
 ∂ₜ V[k₁, k₂] = Lᵥ V[k₁,k₂] - Nᵤ(U[k₁,k₂], V[k₁,k₂])

with Lᵤ U[k₁,k₂]            = -[Dᵤ ( (2πk₁/d)^2 + (2πk₂/d)^2) + F] U[k₁,k₂]
     Nᵤ(U[k₁,k₂], V[k₁,k₂]) = -TF[TF^-1(U) (TF^-1(V))^2]
     Lᵥ V[k₁,k₂]            = -[Dᵥ ( (2πk₁/d)^2 + (2πk₂/d)^2) + (F + k)] V[k₁,k₂]
and we should not forget the term F δ_{k₁,k₂} which introduces a F dt δ_{k₁,k₂} in the integration with dt the time step

we can then use the formulas of Cox and Mathews with the numerical stabilization procedure of Kassam, Trefethen for computing
the terms like (e^z - 1)/z with the Cauchy Integral

References:
 - Notes on FFT-based differentiation, [Johnson, 2011]
 - Fourth-order time stepping for stiff PDEs,  [Kassam, Trefethen, 2005]
'''
class SpectralModel:
    def __init__(self, param_name, N, d=1., dt=0.1):
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
        self.h = d/N
        self.d = d
        self.Du = 2 * 1e-5 / self.h**2
        self.Dv = 1e-5 / self.h**2
        self.dt = dt
        self.noise = 0.2

        self.tf_ut_1 = np.zeros((self.N, self.N), dtype=complex)
        self.tf_vt_1 = np.zeros((self.N, self.N), dtype=complex)
        self.tf_ut = np.zeros((self.N, self.N), dtype=complex)
        self.tf_vt = np.zeros((self.N, self.N), dtype=complex)

        # Precompute various ETDRK4 scalar quantities
        k1, k2 = np.meshgrid(np.arange(self.N).astype(float), np.arange(self.N).astype(float))
        k1[:,self.N/2+1:] -= self.N
        k2[self.N/2+1:,:] -= self.N

        k1 *= 2.0 * np.pi / self.N
        k2 *= 2.0 * np.pi / self.N

        self.Lu = -(self.Du * (k1**2 + k2**2) + self.F)
        self.Lv = -(self.Dv * (k1**2 + k2**2) + self.F + self.k)

        self.E2u = np.exp(self.dt * self.Lu/2.)
        self.Eu = self.E2u ** 2
        
        self.E2v = np.exp(self.dt * self.Lv/2.)
        self.Ev = self.E2v ** 2

        M = 16 # Nb of points for complex means
        r = (np.exp(1j * np.pi * (np.arange(M)+0.5)/M)).reshape((1, M))
        # Generate the points along the unit circle contour over which to compute the mean
        LRu = (self.dt * self.Lu).reshape((self.N*self.N, 1)) + r
        LRv = (self.dt * self.Lv).reshape((self.N*self.N, 1)) + r
        
        # The matrix for integrating the constant F term in the equation of u
        self.F2u = -np.real(np.mean(self.dt * (1. - np.exp(LRu/2.))/LRu, axis=1).reshape((self.N, self.N)))
        self.F2u[1:,:] = 0
        self.F2u[:,1:] = 0
        self.Fu = -np.real(np.mean(self.dt * (1. - np.exp(LRu))/LRu, axis=1).reshape((self.N, self.N)))
        self.Fu[1:,:] = 0
        self.Fu[:,1:] = 0
        
        LRu_2 = LRu**2.
        LRu_3 = LRu**3.
        self.Qu = np.real(np.mean(self.dt * (np.exp(LRu/2.) - 1.) / LRu, axis=1).reshape((self.N, self.N)))
        self.f1u = np.real(np.mean(self.dt * (-4. - LRu + np.exp(LRu) * (4. - 3 * LRu + LRu_2)) / LRu_3, axis=1).reshape((self.N, self.N)))
        self.f2u = np.real(np.mean(self.dt * 2. * (2. + LRu + np.exp(LRu) * (-2. + LRu)) / LRu_3, axis=1).reshape((self.N, self.N)))
        self.f3u = np.real(np.mean(self.dt * (-4. - 3 * LRu - LRu_2 + np.exp(LRu) * (4. - LRu)) / LRu_3, axis=1).reshape((self.N, self.N)))

        LRv_2 = LRv**2.
        LRv_3 = LRv**3.
        self.Qv = np.real(np.mean(self.dt * (np.exp(LRv/2.) - 1.) / LRv, axis=1).reshape((self.N, self.N)))
        self.f1v = np.real(np.mean(self.dt * (-4. - LRv + np.exp(LRv) * (4. - 3 * LRv + LRv_2)) / LRv_3, axis=1).reshape((self.N, self.N)))
        self.f2v = np.real(np.mean(self.dt * 2. * (2. + LRv + np.exp(LRv) * (-2. + LRv)) / LRv_3, axis=1).reshape((self.N, self.N)))
        self.f3v = np.real(np.mean(self.dt * (-4. - 3 * LRv - LRv_2 + np.exp(LRv) * (4. - LRv)) / LRv_3, axis=1).reshape((self.N, self.N)))

    def init(self):
        dN = self.N/4
        
        ut_1 = np.zeros((self.N, self.N), dtype=float)
        ut_1[:,:] = 1
        ut_1[(self.N/2 - dN/2): (self.N/2+dN/2+1), (self.N/2 - dN/2) : (self.N/2+dN/2+1)] = 0.5
        ut_1 += self.noise * (2 * np.random.random((self.N, self.N)) - 1)
        ut_1[ut_1 <= 0] = 0

        vt_1 = np.zeros((self.N, self.N), dtype=float)
        vt_1[:,:] = 0
        vt_1[(self.N/2 - dN/2): (self.N/2+dN/2+1), (self.N/2 - dN/2) : (self.N/2+dN/2+1)] = 0.25
        vt_1 += self.noise * (2 * np.random.random((self.N, self.N)) - 1)
        vt_1[vt_1 <= 0] = 0

        # X, Y = np.meshgrid(np.linspace(-1,1,self.N), np.linspace(-1, 1, self.N))
        # ut_1 = 1. - np.exp(-80 * (( X + .05)**2 + (Y + 0.02)**2))
        # vt_1 = np.exp(-80 * (( X + .05)**2 + (Y + 0.02)**2))
        
        self.tf_ut_1[:,:] = np.fft.fft2(ut_1)
        self.tf_vt_1[:,:] = np.fft.fft2(vt_1)
        self.tf_ut[:,:] = self.tf_ut_1
        self.tf_vt[:,:] = self.tf_vt_1
        
    def get_ut(self):
        return np.real(np.fft.ifft2(self.tf_ut))

        
    def compute_Nuv(self, tf_u, tf_v):
        uv2 = np.fft.fft2(np.fft.ifft2(tf_u).real * (np.fft.ifft2(tf_v).real**2))
        return -uv2, uv2

    def step(self):
        Nu, Nv = self.compute_Nuv(self.tf_ut_1, self.tf_vt_1)
        au = self.E2u * self.tf_ut_1 + self.F2u * self.F *self.N*self.N+ self.Qu * Nu
        av = self.E2v * self.tf_vt_1 + self.Qv * Nv
        Nau, Nav = self.compute_Nuv(au, av)
        bu = self.E2u * self.tf_ut_1 + self.F2u * self.F * self.N * self.N + self.Qu * Nau
        bv = self.E2v * self.tf_vt_1 + self.Qv * Nav
        Nbu, Nbv = self.compute_Nuv(bu, bv)
        cu = self.E2u * au + self.F2u * self.F * self.N * self.N + self.Qu * (2. * Nbu - Nu)
        cv = self.E2v * av + self.Qv * (2. * Nbv - Nv)
        Ncu, Ncv = self.compute_Nuv(cu, cv)

        self.tf_ut[:,:] = self.Eu * self.tf_ut_1 + self.Fu * self.F * self.N * self.N + self.f1u * Nu + self.f2u * (Nau + Nbu) + self.f3u * Ncu
        self.tf_vt[:,:] = self.Ev * self.tf_vt_1 + self.f1v * Nv + self.f2v * (Nav + Nbv) + self.f3v * Ncv
        self.tf_ut_1, self.tf_vt_1 = self.tf_ut, self.tf_vt

        
class Model:

	def __init__(self, param_name, N, mode,d=1.,dt=0.1):
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

                self.vt[:,:] = self.vt_1[:,:]
                self.ut[:,:] = self.ut_1[:,:]

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
    
    N = 100
    pattern = 'worms'
    d = 1.
    dt = 1.

    if(mode <= 2):
        model = Model(pattern, N=N, mode=mode)
    elif mode == 3:
        model = libgrayscott.GrayScott(pattern, N, d, dt)
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



