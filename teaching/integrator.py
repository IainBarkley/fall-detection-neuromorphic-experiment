import nengo
model = nengo.Network()
with model:
    # want to implement: dx/dt = u
    u = nengo.Ensemble(n_neurons=100, dimensions=1)
    stim = nengo.Node(0)
    nengo.Connection(stim, u, synapse=None)
    
    tau_synapse = 0.1
    x = nengo.Ensemble(n_neurons=200, dimensions=1,
                       radius=10)
    
    nengo.Connection(x, x, 
                     transform=1,
                     synapse=tau_synapse,
                     )
                     
                     
    nengo.Connection(u, x,
                     transform=tau_synapse,
                     synapse=tau_synapse)
    