import nengo

model = nengo.Network()
with model:
    stim = nengo.Node([0])
    u = nengo.Ensemble(n_neurons=50, dimensions=1)
    nengo.Connection(stim, u)


    x = nengo.Ensemble(n_neurons=100, dimensions=1,
                    radius=10)
    nengo.Connection(u, x, synapse=0.1)
    
    def recurrent(x):
        return x+0.1
    nengo.Connection(x, x, function=recurrent,
                     synapse=0.01)
