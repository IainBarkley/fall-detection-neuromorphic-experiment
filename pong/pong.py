import nengo
import numpy as np


class Pong(object):
    def __init__(self, dt=0.001, noise=0.5):
        self.ball = np.array([0.5,0])
        self.ball_v = np.array([1, 1])
        self.bounds_x = np.array([0, 1])
        self.bounds_y = np.array([0, 1])
        self.ball_r = 0.05
        self.dt = dt
        self.noise = noise
        
        self.bar = 0.5
        self.bar_width = 0.2
        
    def update(self, t, x):
        bar_v = x[0]
        self.bar += bar_v * self.dt
        if self.bar > self.bounds_x[1]:
            self.bar = self.bounds_x[1]
        if self.bar < self.bounds_x[0]:
            self.bar = self.bounds_x[0]
        
        self.ball += (self.ball_v + np.random.normal(0, self.noise, 2)) * self.dt
        if self.ball[0] > self.bounds_x[1]:
            self.ball[0] = 2*self.bounds_x[1] - self.ball[0]
            self.ball_v[0] *= -1
        if self.ball[0] < self.bounds_x[0]:
            self.ball[0] = 2*self.bounds_x[0] - self.ball[0]
            self.ball_v[0] *= -1
        if self.ball[1] > self.bounds_y[1]:
            self.ball[1] = 2*self.bounds_y[1] - self.ball[1]
            self.ball_v[1] *= -1
        if self.ball[1] < self.bounds_y[0]:
            delta = self.ball[0] - self.bar
            if delta > self.bar_width/2+self.ball_r or delta < -self.bar_width/2-self.ball_r:
                self.ball = np.array([0.5, 1])
                self.ball_v = np.array([np.random.uniform(-1,1), -1])
            else:
                self.ball[1] = 2*self.bounds_y[0] - self.ball[1]
                self.ball_v[1] *= -1
    def make_node(self):
        return nengo.Node(self.update, size_in=1, size_out=0)
        
    def make_display(self):
        def update(t):
            svg = '''
            <svg width="100%" height="100%" viewbox="0 0 100 100">
                <line x1="{x_min}" y1="{y_min}" x2="{x_min}" y2="{y_max}" style="stroke:black"/>
                <line x1="{x_max}" y1="{y_min}" x2="{x_max}" y2="{y_max}" style="stroke:black"/>
                <line x1="{x_min}" y1="{y_max}" x2="{x_max}" y2="{y_max}" style="stroke:black"/>
                <circle cx="{x_ball}" cy="{y_ball}" r="{r_ball}" style="fill:red"/>
                <line x1="{x_min_bar}" y1="{y_min}" x2="{x_max_bar}" y2="{y_min}" style="stroke:blue;stroke-width:5px;"/>
            </svg>
            '''.format(x_min=self.bounds_x[0]*100, x_max=self.bounds_x[1]*100,
                       y_min=100-self.bounds_y[0]*100, y_max=100-self.bounds_y[1]*100,
                       x_ball=self.ball[0]*100, y_ball=100-self.ball[1]*100,
                       r_ball=self.ball_r*100,
                       x_min_bar=100*(self.bar-self.bar_width/2),
                       x_max_bar=100*(self.bar+self.bar_width/2),
                       )
            
            update._nengo_html_ = svg
        return nengo.Node(update)
        
        
        
        
        
pong = Pong()
model = nengo.Network()
with model:
    display = pong.make_display()
    world = pong.make_node()
    
    def keyboard_func(t):
        if 'x' in __page__.keys_pressed:
            return 1
        elif 'z' in __page__.keys_pressed:
            return -1
        else:
            return 0
    keyboard = nengo.Node(keyboard_func)
    
    nengo.Connection(keyboard, world, synapse=None)
    