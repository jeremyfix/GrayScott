# coding: utf-8

import numpy as np
import scipy.signal
import scipy.ndimage
import time
import sys
from threading import Thread, Lock
from time import sleep

# import libgrayscott

'''
We consider a spatial domain of size d × d, with N × N samples

Expressed in the spectral domain, the system we solve is

 ∂ₜ U [k₁,k₂] = -[Dᵤ ( (2πk₁/d)^2 + (2πk₂/d)^2)] U[k₁,k₂]
                - TF[TF^-1(U) (TF^-1(V))^2]
                + F N^2 δ_{k₁,k₂} - F U[k₁,k₂]
 ∂ₜ V [k₁,k₂] = -[Dᵥ ( (2πk₁/d)^2 + (2πk₂/d)^2)] V[k₁,k₂]
                + TF[TF^-1(U) (TF^-1(V))^2] - (F + k) V[k₁,k₂]

with U = TF(u) , V = TF(v)

If we decompose the linear and non-linear parts of the equations,
following the notations of "Fourth order time-stepping for stiff PDEs,
the system reads:

 ∂ₜ U[k₁, k₂] = Lᵤ U[k₁,k₂] + Nᵤ(U[k₁,k₂], V[k₁,k₂]) + F N^2 δ_{k₁,k₂}
 ∂ₜ V[k₁, k₂] = Lᵥ V[k₁,k₂] - Nᵤ(U[k₁,k₂], V[k₁,k₂])

with Lᵤ U[k₁,k₂] = -[Dᵤ ( (2πk₁/d)^2 + (2πk₂/d)^2) + F] U[k₁,k₂]
     Nᵤ(U[k₁,k₂], V[k₁,k₂]) = -TF[TF^-1(U) (TF^-1(V))^2]
     Lᵥ V[k₁,k₂] = -[Dᵥ ( (2πk₁/d)^2 + (2πk₂/d)^2) + (F + k)] V[k₁,k₂]

and we should not forget the term F δ_{k₁,k₂} which introduces
a F dt δ_{k₁,k₂} in the integration with dt the time step

we can then use the formulas of Cox and Mathews with
the numerical stabilization procedure of Kassam, Trefethen
for computing the terms like (e^z - 1)/z with the Cauchy Integral

References:
 - Notes on FFT-based differentiation, [Johnson, 2011]
 - Fourth-order time stepping for stiff PDEs,  [Kassam, Trefethen, 2005]
'''


class SpectralModel:
    ''' Mode can be in ETDFD or ETDRK4 '''
    def __init__(self, param_name, width, height, d=1., dt=0.1, mode='ETDFD'):
        self.param_name = param_name
        if(self.param_name == 'solitons'):
            self.k = 0.056
            self.F = 0.020
        elif(self.param_name == "worms_solitons"):
            self.k = 0.057
            self.F = 0.026
        elif(self.param_name == 'worms'):
            self.k = 0.0630
            self.F = 0.0580
        elif(self.param_name == 'spirals'):
            self.k = 0.050
            self.F = 0.018
        elif(self.param_name == 'exp'):
            self.k = 0.0594
            self.F = 0.0460
        else:
            self.k = 0.040
            self.F = 0.060
        self.width = width
        self.height = height
        self.h = d/self.width
        self.d = d
        self.Du = 2 * 1e-5 / self.h**2
        self.Dv = 1e-5 / self.h**2
        self.dt = dt
        self.noise = 0.2

        # self.tf_ut_1 = np.zeros((self.N, self.N), dtype=complex)
        # self.tf_vt_1 = np.zeros((self.N, self.N), dtype=complex)
        self.cdtype = np.complex64
        self.fdtype = np.float32
        self.tf_ut = np.zeros((self.height, self.width), dtype=self.cdtype)
        self.tf_vt = np.zeros((self.height, self.width), dtype=self.cdtype)

        self.mode = mode
        if self.mode not in ['ETDFD', 'ETDRK4']:
            raise ValueError("mode must be ETDFD or ETDRK4")

        # Precompute various ETDRK4 scalar quantities
        k1, k2 = np.meshgrid(np.arange(self.width).astype(float),
                             np.arange(self.height).astype(self.fdtype))
        k1[:, self.width/2+1:] -= self.width
        k2[self.height/2+1:, :] -= self.height
        k1[:, 0] = 0
        k2[0, :] = 0

        k1 *= 2.0 * np.pi / self.width
        k2 *= 2.0 * np.pi / self.height

        self.Lu = -(self.Du * (k1**2 + k2**2) + self.F)
        self.Lv = -(self.Dv * (k1**2 + k2**2) + self.F + self.k)

        self.E2u = np.exp(self.dt * self.Lu/2.)
        self.Eu = self.E2u ** 2

        self.E2v = np.exp(self.dt * self.Lv/2.)
        self.Ev = self.E2v ** 2

        M = 16  # Nb of points for complex means
        r = (np.exp(1j * np.pi * (np.arange(M)+0.5)/M)).reshape((1, M))
        # Generate the points along the unit circle contour
        # over which to compute the mean
        LRu = (self.dt * self.Lu).reshape((self.width*self.height, 1)) + r
        LRv = (self.dt * self.Lv).reshape((self.width*self.height, 1)) + r

        # The matrix for integrating the constant F term in the equation of u
        self.F2u = -np.real(np.mean(self.dt * (1. - np.exp(LRu/2.))/LRu, axis=1).reshape((self.height, self.width)))
        self.F2u[1:, :] = 0
        self.F2u[:, 1:] = 0
        self.Fu = -np.real(np.mean(self.dt * (1. - np.exp(LRu))/LRu, axis=1).reshape((self.height, self.width)))
        self.Fu[1:, :] = 0
        self.Fu[:, 1:] = 0
        if(mode == 'ETDFD'):
            self.FNu = -np.real(np.mean(self.dt * (1. - np.exp(LRu))/LRu, axis=1).reshape((self.height, self.width)))
            self.FNv = -np.real(np.mean(self.dt * (1. - np.exp(LRv))/LRv, axis=1).reshape((self.height, self.width)))
        elif(mode == 'ETDRK4'):
            LRu_2 = LRu**2.
            LRu_3 = LRu**3.
            self.Qu = np.real(np.mean(self.dt * (np.exp(LRu/2.) - 1.) / LRu, axis=1).reshape((self.height, self.width)))
            self.f1u = np.real(np.mean(self.dt * (-4. - LRu + np.exp(LRu) * (4. - 3 * LRu + LRu_2)) / LRu_3, axis=1).reshape((self.height, self.width)))
            self.f2u = np.real(np.mean(self.dt * 2. * (2. + LRu + np.exp(LRu) * (-2. + LRu)) / LRu_3, axis=1).reshape((self.height, self.width)))
            self.f3u = np.real(np.mean(self.dt * (-4. - 3 * LRu - LRu_2 + np.exp(LRu) * (4. - LRu)) / LRu_3, axis=1).reshape((self.height, self.width)))

            LRv_2 = LRv**2.
            LRv_3 = LRv**3.
            self.Qv = np.real(np.mean(self.dt * (np.exp(LRv/2.) - 1.) / LRv, axis=1).reshape((self.height, self.width)))
            self.f1v = np.real(np.mean(self.dt * (-4. - LRv + np.exp(LRv) * (4. - 3 * LRv + LRv_2)) / LRv_3, axis=1).reshape((self.height, self.width)))
            self.f2v = np.real(np.mean(self.dt * 2. * (2. + LRv + np.exp(LRv) * (-2. + LRv)) / LRv_3, axis=1).reshape((self.height, self.width)))
            self.f3v = np.real(np.mean(self.dt * (-4. - 3 * LRv - LRv_2 + np.exp(LRv) * (4. - LRv)) / LRv_3, axis=1).reshape((self.height, self.width)))

    def init(self):
        dN = min(self.height, self.width)/4

        ut = np.zeros((self.height, self.width), dtype=np.float32)
        ut[:, :] = 1
        ut[(self.height/2 - dN/2): (self.height/2+dN/2+1), (self.width/2 - dN/2) : (self.width/2+dN/2+1)] = 0.5
        ut += self.noise * (2 * np.random.random((self.height, self.width)) - 1)
        ut[ut <= 0] = 0

        vt = np.zeros((self.height, self.width), dtype=float)
        vt[:, :] = 0
        vt[(self.height/2 - dN/2): (self.height/2+dN/2+1), (self.width/2 - dN/2) : (self.width/2+dN/2+1)] = 0.25
        vt += self.noise * (2 * np.random.random((self.height, self.width)) - 1)
        vt[vt <= 0] = 0

        self.tf_ut = np.fft.fft2(ut)
        self.tf_vt = np.fft.fft2(vt)

    def get_ut(self):
        return np.real(np.fft.ifft2(self.tf_ut))

    def get_vt(self):
        return np.real(np.fft.ifft2(self.tf_vt))

    def compute_Nuv(self, tf_u, tf_v):
        uv2 = np.fft.fft2(np.fft.ifft2(tf_u).real * (np.fft.ifft2(tf_v).real**2))
        return -uv2, uv2

    # Erase the reactant in a box
    def erase_reactant(self, center, radius):
        vt = np.real(np.fft.ifft2(self.tf_vt))
        vt[(center[0]-radius):(center[0]+radius), (center[1]-radius):(center[1]+radius)] = 0
        self.tf_vt = np.fft.fft2(vt)

    # Mask the reactant,
    # mask.shape = self.height, self.width
    # mask.dtype = float
    # mask_ij in [0, 1]
    def mask_reactant(self, mask):
        # vt =np.real(np.fft.ifft2(self.tf_vt))
        # vt = vt * mask
        # self.tf_vt = np.fft.fft2(vt)
        vt = np.real(np.fft.ifft2(self.tf_vt))
        vt[mask >= 0.5] = 1.0
        self.tf_vt = np.fft.fft2(vt)

        ut = np.real(np.fft.ifft2(self.tf_ut))
        ut[mask >= 0.5] = 0.0
        self.tf_ut = np.fft.fft2(ut)

    def step(self):
        if(self.mode == 'ETDFD'):
            Nu, Nv = self.compute_Nuv(self.tf_ut, self.tf_vt)
            self.tf_ut = self.Eu * self.tf_ut \
                         + self.Fu * self.F * self.width * self.height\
                         + self.FNu * Nu
            self.tf_vt = self.Ev * self.tf_vt + self.FNv * Nv
        elif(self.mode == 'ETDRK4'):
            Nu, Nv = self.compute_Nuv(self.tf_ut, self.tf_vt)
            au = self.E2u * self.tf_ut + self.F2u * self.F *self.width*self.height+ self.Qu * Nu
            av = self.E2v * self.tf_vt + self.Qv * Nv
            Nau, Nav = self.compute_Nuv(au, av)
            bu = self.E2u * self.tf_ut + self.F2u * self.F * self.width * self.height + self.Qu * Nau
            bv = self.E2v * self.tf_vt + self.Qv * Nav
            Nbu, Nbv = self.compute_Nuv(bu, bv)
            cu = self.E2u * au + self.F2u * self.F * self.width * self.height + self.Qu * (2. * Nbu - Nu)
            cv = self.E2v * av + self.Qv * (2. * Nbv - Nv)
            Ncu, Ncv = self.compute_Nuv(cu, cv)

            self.tf_ut = self.Eu * self.tf_ut + self.Fu * self.F * self.width * self.height + self.f1u * Nu + self.f2u * (Nau + Nbu) + self.f3u * Ncu
            self.tf_vt = self.Ev * self.tf_vt + self.f1v * Nv + self.f2v * (Nav + Nbv) + self.f3v * Ncv


class Model:

    def __init__(self, param_name, width, height, mode,d=1.,dt=0.1):
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
        self.width = width
        self.height = height
        self.h = d/self.width
        self.Du = 2 * 1e-5 / self.h**2
        self.Dv = 1e-5 / self.h**2
        self.dt = dt
        self.noise = 0.2

        self.ut_1 = np.zeros((self.height, self.width), dtype=float)
        self.vt_1 = np.zeros((self.height, self.width), dtype=float)
        self.ut = np.zeros((self.height, self.width), dtype=float)
        self.vt = np.zeros((self.height, self.width), dtype=float)

        self.mode = mode
        if(self.mode == 0):
            self.stencil = np.zeros((self.height, self.width))
            self.stencil[0,0] = -4
            self.stencil[0,1] = 1
            self.stencil[0,-1] = 1
            self.stencil[1,0] = 1
            self.stencil[-1,0] = 1
            self.fft_mask = np.fft.rfft2(self.stencil)
        elif(self.mode == 1):
            self.stencil = np.array([[0, 1., 0], [1., -4., 1.], [0, 1., 0]], dtype=float)

    def init(self):
        dN = min(self.width, self.height)/4
        self.ut_1[:,:] = 1
        self.ut_1[(self.height/2 - dN/2): (self.height/2+dN/2+1), (self.width/2 - dN/2) : (self.width/2+dN/2+1)] = 0.5
        self.ut_1 += self.noise * (2 * np.random.random((self.height, self.width)) - 1)
        self.ut_1[self.ut_1 <= 0] = 0

        self.vt_1[:,:] = 0
        self.vt_1[(self.height/2 - dN/2): (self.height/2+dN/2+1), (self.width/2 - dN/2) : (self.width/2+dN/2+1)] = 0.25
        self.vt_1 += self.noise * (2 * np.random.random((self.height, self.width)) - 1)
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

    def erase_reactant(self, center, radius):
        pass

    def step(self):
        uvv = self.ut_1 * self.vt_1**2
        lu = self.laplacian(self.ut_1)
        lv = self.laplacian(self.vt_1)
        self.ut[:,:] = self.ut_1 + self.dt * (self.Du * lu - uvv + self.F*(1-self.ut_1))
        self.vt[:,:] = self.vt_1 + self.dt * (self.Dv * lv + uvv - (self.F + self.k) * self.vt_1)

        self.ut_1, self.vt_1  = self.ut, self.vt

class ModelOptim:

    def __init__(self, param_name, width, height, d=1.,dt=0.1):
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
        self.width = width+2
        self.height = height+2
        self.h = d/self.width
        self.Du = 2 * 1e-5 / self.h**2
        self.Dv = 1e-5 / self.h**2
        self.dt = dt
        self.noise = 0.2
        self.shape = (self.height, self.width)

        dtype = np.float32
        self.ut_1 = np.zeros(self.shape, dtype=dtype)
        self.vt_1 = np.zeros(self.shape, dtype=dtype)
        self.ut = np.zeros(self.shape, dtype=dtype)
        self.vt = np.zeros(self.shape, dtype=dtype)

    def init(self):
        dN = min(self.width, self.height)/4
        x_ul, y_ul = (self.width/2 - dN/2), (self.height/2 - dN/2)
        x_br, y_br = (self.width/2 + dN/2 + 1), (self.height/2 + dN/2 + 1)

        self.ut_1[:, :] = 1
        self.ut_1[y_ul:y_br, x_ul: x_br] = 0.5
        self.ut_1 += self.noise * (2 * np.random.random(self.shape) - 1)
        self.ut_1[self.ut_1 <= 0] = 0

        self.vt_1[:, :] = 0
        self.vt_1[y_ul:y_br, x_ul:x_br] = 0.25
        self.vt_1 += self.noise * (2 * np.random.random(self.shape) - 1)
        self.vt_1[self.vt_1 <= 0] = 0

        self.vt[:, :] = self.vt_1[:, :]
        self.ut[:, :] = self.ut_1[:, :]

    def laplacian(self, x):
        return -4. * x[1:-1, 1:-1] + \
               (x[1:-1, :-2] + x[:-2, 1:-1] + x[1:-1, 2:] + x[2:, 1:-1])

    def get_ut(self):
        return self.ut[1:-1, 1:-1]

    def erase_reactant(self, center, radius):
        pass

    def step(self):
        uvv = self.ut * self.vt**2
        lu = self.laplacian(self.ut_1)
        lv = self.laplacian(self.vt_1)
        Nu = -uvv + self.F * (1. - self.ut)
        Nv = uvv - (self.F + self.k) * self.vt
        self.ut[1:-1, 1:-1] += self.dt * (self.Du * lu + Nu[1:-1, 1:-1])
        self.vt[1:-1, 1:-1] += self.dt * (self.Dv * lv + Nv[1:-1, 1:-1])


class ThreadedModel(Thread):

    def __init__(self, model):
        super(ThreadedModel, self).__init__()
        self.model = model
        self.mutex = Lock()
        self.running = True
        self.paused = False

    def init(self):
        with self.mutex:
            self.model.init()

    def get_ut(self):
        with self.mutex:
            return self.model.get_ut().copy()

    def mask_reactant(self, mask):
        with self.mutex:
            self.model.mask_reactant(mask)

    def erase_reactant(self, center, radius):
        with self.mutex:
            self.model.erase_reactant(center, radius)

    def keep_running(self):
        with self.mutex:
            return self.running

    def trigger_pause(self):
        with self.mutex:
            self.paused = not self.paused

    def is_paused(self):
        with self.mutex:
            return self.paused

    def run(self):
        while self.keep_running():
            if not self.is_paused():
                self.step()
                sleep(0.000001)
            else:
                sleep(0.1)

    def stop(self):
        with self.mutex:
            self.running = False

    def step(self):
        with self.mutex:
            self.model.step()



def test_basic(model):
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


def test_thread(model):
    model = ThreadedModel(model)

    model.init()

    model.start()

    # Wait in the main thread
    time.sleep(2)
    ut = model.get_ut()

    model.stop()
    model.join()


if(__name__ == '__main__'):

    if(len(sys.argv) <= 1):
        print("Usage : %s mode " % sys.argv[0])
        print("With mode : ")
        print("   0 : spatial model with FFT convolution"
              " in python, forward euler")  # 100 fps
        print("   1 : spatial model with ndimage.convolve"
              " in python, forward euler")  # 165 fps
        print("   2 : spatial model with ndimage.laplace"
              " in python, forward euler")  # 150 fps
        print("   3 : spatial model with fast laplacian "
              "in C++, forward euler")  # 400 fps
        print("   4 : spectral model in python using ETDRK4")
        print("   5 : spatial model, forward Euler")
        sys.exit(-1)

    mode = int(sys.argv[1])

    height = 256
    width = 256
    pattern = 'worms'
    d = 1.
    dt = 1.

    if(mode <= 2):
        model = Model(pattern, width, height, mode=mode)
    elif mode == 3:
        model = None  # libgrayscott.GrayScott(pattern, width, height, d, dt)
    elif mode == 4:
        model = SpectralModel(pattern, height=height, width=width)
    elif mode == 5:
        model = ModelOptim(pattern, width, height)

    # test_basic(model)
    test_thread(model)
