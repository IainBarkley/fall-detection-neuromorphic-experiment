import nengo

model = nengo.Network()
with model:
    stim = nengo.Node([0])
    a = nengo.Ensemble(n_neurons=50, dimensions=1,
                        neuron_type=nengo.LIF(),
                        radius=2)
    nengo.Connection(stim, a)
    
    b = nengo.Ensemble(n_neurons=75, dimensions=1,
                        radius=4)
    
    
    def my_function(x):
        return x**2
    nengo.Connection(a, b, function=my_function)
