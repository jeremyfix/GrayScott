# coding: utf-8

import numpy as np
import scipy.signal
import scipy.ndimage
import numpy as np
import time
import sys


''' 
We consider a spatial domain of size d × d, with N × N samples

Expressed in the spectral domain, the system we solve is 

OUTDATED !!!!!

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
    ''' Mode can be in ETDFD or ETDRK4 '''
    def __init__(self, param_name, width, height, d=1., dt=0.1, mode='ETDFD'):
        self.param_name = param_name
        if(self.param_name == 'labyrinth'):
            # self.a0 = -0.1
            # self.a1 = 2
            # self.epsilon = 0.05
            # self.delta = 4.
            self.Ku = 1e-4
            self.a = 0.1
            self.epsilon = 0.01
            self.beta = 0.5
            self.gamma = 1.
        else:
            raise Exception("Unknown parameters")
        self.width = width
        self.height = height
        #self.h = d/self.width
        self.d = d
        self.dt = dt
        self.noise = 0.2

        self.cdtype = np.complex64
        self.fdtype = np.float32
        self.tf = np.zeros((2, self.height, self.width), dtype=self.cdtype)

        self.mode = mode
        if(not self.mode in ['ETDFD']):
            print("The numerical scheme you mentioned is not implemented")
            raise Exception("Unknown numerical scheme, must be ETDFD")

        # Precompute various ETDRK4 scalar quantities
        k1, k2 = np.meshgrid(np.arange(self.width).astype(float), np.arange(self.height).astype(self.fdtype))
        k1[:,self.width/2+1:] -= self.width
        k2[self.height/2+1:,:] -= self.height

        k1 *= 2.0 * np.pi / self.width
        k2 *= 2.0 * np.pi / self.height

        k = -(k1**2 + k2**2)
        
        
        self.E = np.zeros((self.height, self.width, 2, 2))
        self.FN = np.zeros((self.height, self.width, 2, 2))
        for i in range(self.height):
            for j in range(self.width):
                Luv2x2 = np.zeros((2, 2))
                # Luv2x2[0, 0] = 1 + k[i, j]
                # Luv2x2[0, 1] = -1
                # Luv2x2[1, 0] = self.epsilon
                # Luv2x2[1, 1] = -self.epsilon * self.a1 + self.delta * k[i,j]
                Luv2x2[0, 0] = -self.a + self.Ku *  k[i, j]
                Luv2x2[0, 1] = -1
                Luv2x2[1, 0] = self.epsilon * self.beta
                Luv2x2[1, 1] = -self.epsilon * self.gamma

                
                self.E[i, j, :, :] = np.exp(self.dt * Luv2x2)
                self.FN[i, j, :, :] = np.dot(np.linalg.inv(Luv2x2), self.E[i, j, :, :] - np.eye(2))
    
    def init(self):
        dN = self.width/32.
        
        ut = np.zeros((self.height, self.width), dtype=np.float32)
        ut[:, (self.width/2-dN):(self.width/2+dN)] = 1.
        ut[ut <= 0] = 0
        for i in range(self.height):
            shift = int(5*np.exp(-(i - self.height/2.)**2/(2.*10.**2))*np.cos(i*2.*np.pi/20) + (2.0 * np.random.random() - 1.)* 3.)
            ut[i,:] = np.roll(ut[i,:], shift)

        vt = np.zeros((self.height, self.width), dtype=float)

        self.tf[0, :, :] = np.fft.fft2(ut)
        self.tf[1, :, :] = np.fft.fft2(vt)
       
    def get_ut(self):
        return np.real(np.fft.ifft2(self.tf[0, :, :]))
    
    def get_vt(self):
        return np.real(np.fft.ifft2(self.tf[1, :, :]))

    
    def compute_Nuv(self):
        u = np.fft.ifft2(self.tf[0, :, :]).real
        Nu = (1. + self.a) * u**2 - u**3
        Nv = np.zeros((self.height, self.width))#-self.epsilon * self.delta * np.fft.fft2(np.ones((self.height, self.width)))
        
        #Nu = np.fft.fft2(u**3)
        #Nv = -self.epsilon * self.a0 * np.fft.fft2(np.ones((self.height, self.width)))
        return np.stack((Nu, Nv), axis=0)
      
    def step(self):
        u = self.get_ut()
        print("bef %f %f"%(u.min(), u.max()))
        Nuv = self.compute_Nuv()
        for i in range(self.height):
            for j in range(self.width):
                self.tf[:, i, j] = np.dot(self.E[i, j, :, :], self.tf[:,i, j]) + np.dot(self.FN[i, j, :, :], Nuv[:, i, j])

        
class Model:

    def __init__(self, param_name, width, height,d=1.,dt=0.1):
        self.param_name = param_name
        if(self.param_name == 'labyrinth'):
            self.a0 = -0.1
            self.a1 = 2
            self.epsilon = 0.05
            self.delta = 4.
        self.width = width
        self.height = height
        self.h = d/self.width
        self.dt = dt
        self.noise = 0.2
        
        self.ut_1 = np.zeros((self.height, self.width), dtype=float)
        self.vt_1 = np.zeros((self.height, self.width), dtype=float)
        self.ut = np.zeros((self.height, self.width), dtype=float)
        self.vt = np.zeros((self.height, self.width), dtype=float)
        
        self.stencil = np.array([[0, 1., 0], [1., -4., 1.], [0, 1., 0]], dtype=float)

    def init(self):
        dN = self.width/32
        self.ut_1[:,:] = 0
        self.ut_1[:, (self.width/2-dN):(self.width/2+dN)] = 1
        #self.ut_1 += self.noise * (2 * np.random.random((self.height, self.width)) - 1)
        self.ut_1[self.ut_1 <= 0] = 0
        for i in range(self.height):
            shift = int(5*np.exp(-(i - self.height/2.)**2/(2.*10.**2))*np.cos(i*2.*np.pi/20) + (2.0 * np.random.random() - 1.)* 3.)
            self.ut_1[i,:] = np.roll(self.ut_1[i,:], shift)

        
        self.vt_1[:,:] = 0        
        
        self.vt[:,:] = 0
        self.ut[:,:] = self.ut_1[:,:]
        
    def laplacian(self, x):
        return scipy.ndimage.convolve(x, self.stencil, mode='wrap')
    
    def get_ut(self):
        return self.ut

    def erase_reactant(self, center, radius):
        pass

    def step(self):
        lu = self.laplacian(self.ut_1)
        lv = self.laplacian(self.vt_1)	
        self.ut[:,:] = self.ut_1 + self.dt * (lu + self.ut_1 - self.ut_1**3 - self.vt_1)
        self.vt[:,:] = self.vt_1 + self.dt * (self.delta * lv + self.epsilon * (self.ut_1 - self.a1 * self.vt_1 - self.a0))
        
        self.ut_1, self.vt_1  = self.ut, self.vt

            

if(__name__ == '__main__'):

    if(len(sys.argv) <= 1):
        print("Usage : %s mode "% sys.argv[0])
        print("With mode : ")
        print("   0 : spatial model with ndimage.convolve in python, forward euler") # 165 fps
        print("   1 : spectral model in python using ETDRK4")

        sys.exit(-1)

    mode = int(sys.argv[1])
    
    height = 128
    width = 128
    pattern = 'labyrinth'
    d = 1.
    dt = 0.001

    if(mode == 0):
        model = Model(pattern, width, height)
    elif mode == 1:
        model = SpectralModel(pattern, height=height, width=width)

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



