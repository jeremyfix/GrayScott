# GrayScott simulation using ETDRK4

These scripts simulate the GrayScott reaction diffusion system. To speed up the simulation, the simulation is performed in the spectral domain, implementing the Exponential Time Difference fourth order numerical scheme of [Kassam et al(2005)].

If you wish to play to with the system, just run the cv2_grayscott.py script. There are some post-processing to make the images look nicer.

There is also a script for using the Kinect input in order to play with the reaction/diffusion system.

The parameters of the equation for some of the patterns are already provided.

![Solitons](https://raw.githubusercontent.com/jeremyfix/GrayScott/master/u-solitons.png)

![Worms](https://raw.githubusercontent.com/jeremyfix/GrayScott/master/u-worms.png)

![Spirals](https://raw.githubusercontent.com/jeremyfix/GrayScott/master/u-spirals.png)

Related papers:
- Complex Patterns in a Simple System, J.E. Pearson, Science(261), 189--192
- A.-K. Kassam and L. N. Trefethen (2005), Fourth-order time-stepping for stiff PDEs, SIAM J. Sci. Comput. 26, 1214–1233.
- H. Montanelli and N. Bootland (2016) Solving periodic semilinear stiff pdes in 1d, 2d and 3d with exponential integrators


Interesting links : 
- http://mrob.com/pub/comp/xmorphia/
