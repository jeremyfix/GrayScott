#from vispy import app
#from vispy.gloo import clear, set_clear_color, set_viewport, Program

from vispy.util.transforms import ortho
from vispy import gloo
from vispy import app

import numpy as np
import libgrayscott


VERT_SHADER = """
// Uniforms
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_antialias;

// Attributes
attribute vec2 a_position;
attribute vec2 a_texcoord;

// Varyings
varying vec2 v_texcoord;

// Main
void main (void)
{
    v_texcoord = a_texcoord;
    gl_Position = u_projection * u_view * u_model * vec4(a_position,0.0,1.0);
}
"""

FRAG_SHADER = """
uniform sampler2D u_texture;
varying vec2 v_texcoord;
void main()
{
    gl_FragColor = texture2D(u_texture, v_texcoord);
    gl_FragColor.a = 1.0;
}

"""

class Canvas(app.Canvas):

    def __init__(self, N, pattern):
        app.Canvas.__init__(self, keys='interactive', size=(512, 512))

        self.N = N
        self.pattern = pattern
        self.gs_model = libgrayscott.GrayScott(self.pattern, self.N)
        self.u = np.random.random((N, N))
        self.gs_model.init()

        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.texture = gloo.Texture2D(self.u, interpolation='linear')

	# A simple texture quad
	self.data = np.zeros(4, dtype=[('a_position', np.float32, 2),
                          ('a_texcoord', np.float32, 2)])
	self.data['a_position'] = np.array([[0, 0], [N, 0], [0, N], [N, N]])
	self.data['a_texcoord'] = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])



        self.program['u_texture'] = self.texture
        self.program.bind(gloo.VertexBuffer(self.data))

        self.view = np.eye(4, dtype=np.float32)
        self.model = np.eye(4, dtype=np.float32)
        self.projection = np.eye(4, dtype=np.float32)

        self.program['u_model'] = self.model
        self.program['u_view'] = self.view
        self.projection = ortho(0, N, 0, N, -1, 1)
        self.program['u_projection'] = self.projection

        gloo.set_clear_color('black')

        self._timer = app.Timer(1/10., connect=self.update, start=True)
	self._timer_compute = app.Timer(1/100., connect=self.on_step, start=True)

        self.show()

    def on_step(self, event):
	self.u = self.gs_model.step()

    def on_draw(self, event):
        gloo.clear(color=True, depth=True)
        self.texture.set_data(self.u.astype('float32'))
        self.program.draw('triangle_strip')

    def on_resize(self, event):
        width, height = event.physical_size
        gloo.set_viewport(0, 0, width, height)
        self.projection = ortho(0, width, 0, height, -100, 100)
        self.program['u_projection'] = self.projection

        # Compute thje new size of the quad
        r = width / float(height)
        R = self.N / float(self.N)
        if r < R:
            w, h = width, width / R
            x, y = 0, int((height - h) / 2)
        else:
            w, h = height * R, height
            x, y = int((width - w) / 2), 0
        self.data['a_position'] = np.array(
            [[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
        self.program.bind(gloo.VertexBuffer(self.data))



#class Canvas(app.Canvas):
#    def __init__(self, N, pattern):
#
#
#	self.N = N
#	self.pattern = pattern
#	self.model = libgrayscott.GrayScott(self.pattern, self.N)
#	self.u = np.random.random((N, N))
#	self.model.init()
#
#        app.Canvas.__init__(self, size=(512, 512),
#                            keys='interactive')
#
#	self._timer_draw = app.Timer('auto', connect=self.on_draw, start=True)
#	self._timer_compute = app.Timer(1/100., connect=self.on_step, start=True)
#
#        self.image = Program(img_vertex, img_fragment, 4)
#        self.image['position'] = (-1, -1), (-1, +1), (+1, -1), (+1, +1)
#        self.image['texcoord'] = (0, 0), (0, +1), (+1, 0), (+1, +1)
#        self.image['vmin'] = +0.0
#        self.image['vmax'] = +1.0
#        self.image['cmap'] = 0  # Colormap index to use
#
#        self.image['colormaps'] = colormaps
#        self.image['colormaps'].interpolation = 'linear'
#        self.image['colormaps_shape'] = colormaps.shape[1], colormaps.shape[0]
#
#        set_clear_color('black')
#
#        self.show()
#
#    def on_resize(self, event):
#        width, height = event.physical_size
#        set_viewport(0, 0, *event.physical_size)
#
#    def on_step(self, event):
#	self.model.init()
#	self.u = self.model.step()
#
#    def on_draw(self, event):
#	self.image['image'] = self.u.astype('float32')
#        self.image['image'].interpolation = 'linear'
#
#        clear(color=True, depth=True)
#        self.image.draw('triangle_strip')

if __name__ == '__main__':
    canvas = Canvas(256, 'uskate')
    app.run()


