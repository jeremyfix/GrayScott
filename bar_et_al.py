import cv2
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt


#Bär et al 1993 : Spiral waves in a surface reaction: Model calculations
# The system of Bar et al
@np.vectorize
def delayed_inhibitor_production(u):
    if u <= 1./3.:
        return 0.0
    elif u < 1:
        return 1. - 6.75 * u * (u-1)**2
    else:
        return 1.
 
def laplacian(u):
    stencil = np.array([[0, 1, 0],[1, -4, 1], [0, 1, 0]])
    return scipy.ndimage.convolve(u, stencil, mode='constant')

def step_bar(u, w, epsilon, a, b, dt):
    return u + dt * (-1./epsilon * u * (u-1) * (u - (w + b)/a) + laplacian(u)) + np.random.random(u.shape)*0.001,  w + dt * (delayed_inhibitor_production(u) - w )



N = 200
epsilon = 0.025
a = 0.84
b = 0.105
dt = 0.05

u = np.random.random((N, N))*0.2
w = np.random.random((N, N))*0.2

#plt.figure()
#plt.imshow(u)

cv2.namedWindow("u")
cv2.namedWindow("w")

key=0
while key != ord('q'):
    u, w = step_bar(u, w, epsilon, a, b, dt)

    print(w.max())
    #w_color = cv2.applyColorMap((255*w).astype(int), cv2.COLORMAP_HOT)
    #print(w_color.shape)

    cv2.imshow("u",u/u.max())
    cv2.imshow("w",w)
    key = cv2.waitKey(1) & 0xFF

