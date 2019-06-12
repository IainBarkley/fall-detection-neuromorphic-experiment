import nengo
model = nengo.Network()
with model:
    # want to implement: dx/dt = (u-x)/tau_desired
    u = nengo.Ensemble(n_neurons=100, dimensions=1)
    stim = nengo.Node(0)
    nengo.Connection(stim, u, synapse=None)
    
    tau_desired = 0.001
    tau_synapse = 0.2
    x = nengo.Ensemble(n_neurons=500, dimensions=1)
    
    #def recurrent(x):
        # tau_syn * f(x) + x
        #return tau_synapse*(-x/tau_desired) + x
    #    return (1-tau_synapse/tau_desired)*x
    nengo.Connection(x, x, 
                     #function=recurrent,
                     transform=(1-tau_synapse/tau_desired),
                     synapse=tau_synapse)
                     
    #def feedforward(u):
    #    return tau_synapse*(u/tau_desired)
    nengo.Connection(u, x,
                     transform=tau_synapse/tau_desired, 
                     #function=feedforward,
                     synapse=tau_synapse)
    