import nengo
model = nengo.Network()
with model:
    stim_a = nengo.Node(0)
    a = nengo.Ensemble(n_neurons=100, dimensions=1)
    nengo.Connection(stim_a, a)
    
    stim_b = nengo.Node(0)
    b = nengo.Ensemble(n_neurons=100, dimensions=1)
    nengo.Connection(stim_b, b)
    
    c = nengo.Ensemble(n_neurons=100, dimensions=2, radius=1.5)
    
    def func_1(x):
        return x,0
    nengo.Connection(a, c, function=func_1)
    #nengo.Connection(a, c[0])

    def func_2(x):
        return 0,x
    nengo.Connection(b, c, function=func_2)
    #nengo.Connection(b, c[1])
    
    d = nengo.Ensemble(n_neurons=100, dimensions=1,
                       #encoders=[[1]], max_rates=[100],
                       #intercepts=[-0.8]
                       )
    def product(x):
        return x[0]*x[1]
    nengo.Connection(c, d, function=product)
    