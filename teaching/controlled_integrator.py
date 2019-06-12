import nengo

model = nengo.Network()
with model:
    stim_u = nengo.Node([0])
    u = nengo.Ensemble(n_neurons=100, dimensions=1)
    nengo.Connection(stim_u, u)
    
    stim_alpha = nengo.Node(0)
    alpha = nengo.Ensemble(n_neurons=100, dimensions=1)
    nengo.Connection(stim_alpha, alpha)
    
    tau_synapse = 0.1
    integrator = nengo.Ensemble(n_neurons=1000, dimensions=2,
                                radius=1.5)
    nengo.Connection(alpha, integrator[0])
    nengo.Connection(u, integrator[1], transform=tau_synapse)
    def function(x):
        return x[0]*x[1]
    nengo.Connection(integrator, integrator[1], 
                     function=function,
                     synapse=tau_synapse)