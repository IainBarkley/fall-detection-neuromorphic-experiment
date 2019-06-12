import nengo
import numpy as np
from shapes import circle, rectangle, triangle, canvas


class Environment(object):

    def __init__(self, size=10, dt=0.001):

        self.size = size
        self.x = size / 2.
        self.y = size / 2.
        self.th = 0

        self.dt = dt

        self._nengo_html_ = ''

    def __call__(self, t, v):

        self.th += v[1] * self.dt
        self.x += np.cos(self.th)*v[0]*self.dt
        self.y += np.sin(self.th)*v[0]*self.dt

        self.x = np.clip(self.x, 0, self.size)
        self.y = np.clip(self.y, 0, self.size)
        if self.th > np.pi:
            self.th -= 2*np.pi
        if self.th < -np.pi:
            self.th += 2*np.pi

        direction = self.th * 180. / np.pi + 90.
        
        shape_list = [
            # bounding box
            rectangle(
                width=self.size, height=self.size, color='white',
                outline_color='black', outline_width=0.1,
            ),
            # agent
            triangle(x=self.x, y=self.y, th=direction, color='blue'),
        ]
        
        # draw all of the shapes on the screen
        self._nengo_html_ = canvas(shape_list, width=self.size, height=self.size)

        return self.x, self.y, self.th

model = nengo.Network(seed=13)

with model:

    env = nengo.Node(
        Environment(
            size=10,
        ),
        size_in=2,
        size_out=3,
    )

    velocity_input = nengo.Node([0, 0])

    nengo.Connection(velocity_input, env)
