import nengo

model = nengo.Network()
with model:
    stim = nengo.Node([0])
    a = nengo.Ensemble(n_neurons=500, dimensions=1,
                        neuron_type=nengo.LIFRate(),
                        radius=1)
    nengo.Connection(stim, a)
    
    b = nengo.Ensemble(n_neurons=75, dimensions=1,
                        radius=1, 
                        neuron_type=nengo.LIFRate())
    
    
    def my_function(x):
        if x<0:
            return -1
        else:
            return 1
    nengo.Connection(a, b, function=my_function,
                      synapse=None)
