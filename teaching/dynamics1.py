import nengo
model = nengo.Network()
with model:
    ens = nengo.Ensemble(n_neurons=100, dimensions=1)
    stim = nengo.Node(0)
    nengo.Connection(stim, ens, synapse=0.005)
    
    ens2 = nengo.Ensemble(n_neurons=100, dimensions=1)
    def my_function(x):
        return x**2
    nengo.Connection(ens, ens2, function=my_function,
                     synapse=0.2)