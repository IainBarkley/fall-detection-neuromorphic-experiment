import nengo
import numpy as np

model = nengo.Network()
with model:
    stim = nengo.Node([0,0])
    a = nengo.Ensemble(n_neurons=1000, dimensions=2, radius=1.5)
    nengo.Connection(stim, a, synapse=0.01)
    
    output = nengo.Node(None, size_in=1)
    
    def product(x):
        return x[0]*x[1]
        
    inputs = [[1,1], [1,-1], [-1,1], [-1,-1]]
    outputs = [[1], [-1], [-1], [1]]
    
#    conn = nengo.Connection(a, output, function=product)
    conn = nengo.Connection(a, output, eval_points=inputs,
                            function=outputs)
                     
