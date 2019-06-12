import nengo
model = nengo.Network()
with model:
    ens = nengo.Ensemble(n_neurons=100, dimensions=1)
    stim = nengo.Node(0)
    nengo.Connection(stim, ens, synapse=0.005)
    
    ens2 = nengo.Ensemble(n_neurons=100, dimensions=2)
    #def func1(x):
    #    return x, 0
    #nengo.Connection(ens, ens2, function=func1)
    nengo.Connection(ens, ens2[0])
    ens3 = nengo.Ensemble(n_neurons=100, dimensions=1)
    stim2 = nengo.Node(0)
    nengo.Connection(stim2, ens3, synapse=0.005)

    #def func2(x):
    #    return 0, x
    #nengo.Connection(ens3, ens2, function=func2)   
    nengo.Connection(ens3, ens2[1])
    