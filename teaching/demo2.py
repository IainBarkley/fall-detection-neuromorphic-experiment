import nengo
import numpy as np

model = nengo.Network()
with model:
    def stim_func(t):
        return np.sin(2*np.pi*t)
    
    stim = nengo.Node(stim_func)
    a = nengo.Ensemble(n_neurons=100, dimensions=1)
    nengo.Connection(stim, a, synapse=0.01)
    
    output = nengo.Node(None, size_in=1)
    conn = nengo.Connection(a.neurons, output,
                            transform=np.zeros((1,100)),
                            learning_rule_type=nengo.PES(
                                learning_rate=0.01),
                            synapse=0.001)
                     
    error = nengo.Node(None, size_in=1)
    nengo.Connection(output, error, transform=1, synapse=None)
    nengo.Connection(stim, error, transform=-1, synapse=None)
    nengo.Connection(error, conn.learning_rule, synapse=None)
