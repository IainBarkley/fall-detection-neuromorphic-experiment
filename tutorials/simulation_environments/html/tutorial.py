import nengo
import numpy as np
from shapes import canvas,rectangle, circle,triangle
import math

class Environment:
    def __init__(self,size=10):
        self.size = 10
        self.x = self.size / 2
        self.y = self.size / 2
        self.th = 0.
        self.dt = 0.001
        self._nengo_html_ = ""

    # takes time, and input
    def __call__(self,t,x):
        
        self.x = np.clip( self.x + self.dt * x[0], 0., self.size )
        self.y = np.clip( self.y + self.dt * x[1], 0., self.size )
        self.th = self.th + self.dt * x[1]
        direction = self.th * 180 / math.pi + 90
        
        shape_list = [
                rectangle(height=self.size,width=self.size),
                triangle(x=self.x,y=self.y,th=direction),
                ]
        
        self._nengo_html_ = canvas(shape_list=shape_list)

model = nengo.Network()
with model:
    
    env = nengo.Node(Environment(),size_in=2,size_out=0)
    position = nengo.Node([0,0])
    nengo.Connection(position,env)