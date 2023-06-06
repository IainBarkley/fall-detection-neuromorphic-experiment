import nengo
import numpy as np

model = nengo.Network()
with model:
    stim = nengo.Node([0,0])
    a = nengo.Ensemble(n_neurons=500, dimensions=2,
                        neuron_type=nengo.LIFRate(),
                        radius=1)
    nengo.Connection(stim, a)
    
    b = nengo.Ensemble(n_neurons=75, dimensions=1,
                        radius=1, 
                        neuron_type=nengo.LIFRate())
    
    inputs = [[1,1],[1,0],[0,1],[0,0]]
    outputs = [[0],[1],[1],[0]]
    
    nengo.Connection(a, b, eval_points=inputs,
                       function=outputs,
                      synapse=None)
