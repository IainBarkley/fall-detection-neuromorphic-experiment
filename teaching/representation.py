import nengo
model = nengo.Network()
with model:
    ens = nengo.Ensemble(n_neurons=1000, dimensions=2,
                         intercepts=nengo.dists.Uniform(0.5,1),
                         radius=2)
    
    stim1 = nengo.Node(0)
    stim2 = nengo.Node(0)
    nengo.Connection(stim1, ens[0])
    nengo.Connection(stim2, ens[1])
    
    
    
    
    