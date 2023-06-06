import nengo
import numpy as np



model = nengo.Network()
with model:
    def pulse(t):
        if t<0.2:
            return 1
        else:
            return 0
    stim = nengo.Node(pulse)

    speed = nengo.Node(0)

    osc = nengo.Ensemble(n_neurons=1000, 
                   dimensions=3, radius=1.5)

    

    nengo.Connection(speed, osc[2])
    nengo.Connection(stim, osc[0])

    synapse = 0.1

    def feedback(x):
        omega = 10
        return (omega*synapse*x[1]*x[2] + 1*x[0],
                -omega*synapse*x[0]*x[2] + 1*x[1])

    nengo.Connection(osc, osc[:2], 
                 synapse=synapse, function=feedback)

    heart = nengo.Ensemble(n_neurons=500, dimensions=2,
                    radius=3)

    

    def heart_func(x):
        theta = np.arctan2(x[1], x[0])
        r0 = np.sqrt(x[0]**2 + x[1]**2)

        r = (2 - 2 * np.sin(theta) + 
             np.sin(theta) * (np.sqrt(np.abs(np.cos(theta)))/
                              (np.sin(theta)+1.4)))

        return r0*r*np.cos(theta), r0*r*np.sin(theta)

        

    nengo.Connection(osc, heart, function=heart_func)