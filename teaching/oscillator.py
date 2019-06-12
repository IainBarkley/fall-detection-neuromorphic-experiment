import nengo
model = nengo.Network()
with model:
    # dx/dt = [[0,w],[-w,0]]*x
    
    # feedback func:   tau*[[0,w],[-w,0]]*x + x
    # feedback func:   tau*[[0,w],[-w,0]]*x + [[1,0],[0,1]]*x
    #                  [[1,tau*w],[-tau*w,1]]*x 
    osc = nengo.Ensemble(n_neurons=200, dimensions=2)
    
    tau = 0.1
    w = 1
    def feedback(x):
        x0, x1 = x
        return x0+tau*w*x1, -tau*w*x0+x1
    nengo.Connection(osc, osc, function=feedback, synapse=tau)
    
    #def stim_func(t):
    #    if t < 0.1:
    #        return 1
    #    else:
    #        return 0
    #stim = nengo.Node(stim_func)
    stim = nengo.Node(lambda t: 1 if t<0.1 else 0)
    nengo.Connection(stim, osc[0])
    
    
        
        
        