import nengo
import numpy as np

model = nengo.Network()
#model.config[nengo.Connection].synapse=None
with model:
    stim = nengo.Node([0,0])
    a = nengo.Ensemble(n_neurons=500, dimensions=2,
                        neuron_type=nengo.LIFRate(),
                        radius=1)
    nengo.Connection(stim, a)
    
    b = nengo.Ensemble(n_neurons=75, dimensions=1,
                        radius=1, 
                        neuron_type=nengo.LIFRate())
    
    error = nengo.Ensemble(n_neurons=50, dimensions=1)
    
    
    def zero(x):
        return 0
    conn = nengo.Connection(a, b, function=zero,
                     learning_rule_type=nengo.PES(),
                      synapse=None)
                      
    nengo.Connection(b, error, transform=1)
    
    def correct(x):
        return x[0]-x[1]
    nengo.Connection(stim, error, function=correct,
                     transform=-1)
    nengo.Connection(error, conn.learning_rule)  
    
    def stop_learning_func(t):
        if t>10:
            return 1
        else:
            return 0
    stop_learning = nengo.Node(stop_learning_func)
    nengo.Connection(stop_learning, error.neurons,
                     transform=-3*np.ones((50,1)))
