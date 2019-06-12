import nengo
import numpy as np
model = nengo.Network()
with model:
    
    ens = nengo.Ensemble(n_neurons=1000, dimensions=2,
                         radius=1.5)
    
    def feedback(x):
        theta = np.arctan2(x[1], x[0])
        theta = theta + 0.05
        x = np.cos(theta)
        y = np.sin(theta)
        if x > abs(y):
            y = y / x
            x = 1
        elif x < -abs(y):
            y = -y / x
            x = -1
        elif y > abs(x):
            x = x / y
            y = 1
        else:
            x = -x / y
            y = -1
        return x, y
    nengo.Connection(ens, ens, function=feedback)