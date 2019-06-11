import nengo
import numpy as np

model = nengo.Network()
with model:
    ens = nengo.Ensemble(n_neurons=100, dimensions=1,
                         radius=1)
                         
    ens.radius=5
    
    stim = nengo.Node(np.sin)
    nengo.Connection(stim, ens)

    value = nengo.Node(None, size_in=1)
    
    conn = nengo.Connection(ens.neurons, value,
                     transform=[[0]*100],
                     learning_rule_type=nengo.PES(learning_rate=1e-5))
                     
    error = nengo.Node(None, size_in=1)
    nengo.Connection(value, error)
    nengo.Connection(stim, error, transform=-1)
    nengo.Connection(error, conn.learning_rule)
    

    
    
    