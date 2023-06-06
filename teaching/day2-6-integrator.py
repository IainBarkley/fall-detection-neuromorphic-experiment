import nengo

model = nengo.Network()
with model:
    stim = nengo.Node([0])
    u = nengo.Ensemble(n_neurons=50, dimensions=1)
    nengo.Connection(stim, u)

    tau_actual = 0.1
    
    def forward(u):
        return tau_actual*u

    x = nengo.Ensemble(n_neurons=500, dimensions=1,
                    radius=1)
    nengo.Connection(u, x, synapse=tau_actual,
                     function=forward)
    
    def recurrent(x):
        return x
    nengo.Connection(x, x, function=recurrent,
                     synapse=tau_actual)
