# Simple agent that moves towards a circle controlled by sliders

import nengo
import numpy as np

class Environment(object):

    def __init__(self, size=10, dt=0.001):

        self.size = size
        self.agent_x = size / 2.
        self.agent_y = size / 2.
        self.target_x = size / 2.
        self.target_y = size / 2.

        self.dt = dt

        self._nengo_html_ = ''
        
        self.circle_template = '<circle cx="{0}" cy="{1}" r=".3" fill="{2}"/>'

    def _generate_svg(self):
        
        # Header for the svg
        svg = '<svg width="100%%" height="100%%" viewbox="0 0 {0} {1}">'.format(self.size, self.size)
        
        # Draw the outside bounding box
        svg += '<rect width="{0}" height="{1}" style="fill:white;stroke:black;stroke-width:.1"/>'.format(self.size, self.size)

        # Draw the agent
        svg += self.circle_template.format(self.agent_x, self.agent_y, 'blue')

        # Draw the target
        svg += self.circle_template.format(self.target_x, self.target_y, 'green')
        
        # Closing tag
        svg += '</svg>'

        self._nengo_html_ = svg

    def scale(self, v):

        return (v / self.size) * 2 - 1

    def __call__(self, t, v):

        self.agent_x += v[0]*self.dt
        self.agent_y += v[1]*self.dt

        self.target_x = v[2]
        self.target_y = v[3]

        self.agent_x = np.clip(self.agent_x, 0, self.size)
        self.agent_y = np.clip(self.agent_y, 0, self.size)
        self.target_x = np.clip(self.target_x, 0, self.size)
        self.target_y = np.clip(self.target_y, 0, self.size)

        self._generate_svg()

        # Normalize between -1 and 1 when returning
        return [self.scale(self.agent_x),
                self.scale(self.agent_y),
                self.scale(self.target_x),
                self.scale(self.target_y)]
        #return self.agent_x, self.agent_y, self.target_x, self.target_y

def control(v, vel=3):
    dx = v[2] - v[0]
    dy = v[3] - v[1]

    return vel * dx, vel * dy

model = nengo.Network(seed=13)

with model:

    target_position = nengo.Node([0, 0])

    state = nengo.Ensemble(n_neurons=400, dimensions=4, radius=1)

    env = nengo.Node(
        Environment(
            size=10,
        ),
        size_in=4,
        size_out=4,
    )

    #velocity_input = nengo.Node([0, 0])

    #nengo.Connection(velocity_input, env[[0, 1]])
    nengo.Connection(target_position, env[[2, 3]])
    nengo.Connection(env, state)

    nengo.Connection(state, env[[0, 1]], function=control)
