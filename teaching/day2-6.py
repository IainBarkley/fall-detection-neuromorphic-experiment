import nengo

model = nengo.Network()
with model:
    stim = nengo.Node([0])
    u = nengo.Ensemble(n_neurons=50, dimensions=1)
    nengo.Connection(stim, u)

    tau_desired = 0.01
    tau_actual = 0.1
    
    def forward(u):
        return tau_actual * (u/tau_desired)

    x = nengo.Ensemble(n_neurons=500, dimensions=1,
                    radius=1)
    nengo.Connection(u, x, synapse=tau_actual,
                     function=forward)
    
    def recurrent(x):
        return -tau_actual*x/tau_desired + x
    nengo.Connection(x, x, function=recurrent,
                     synapse=tau_actual)
