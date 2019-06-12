import nengo
model = nengo.Network()
with model:
    # dx/dt = [[0,w],[-w,0]]*x
    
    # feedback func:   tau*[[0,w],[-w,0]]*x + x
    # feedback func:   tau*[[0,w],[-w,0]]*x + [[1,0],[0,1]]*x
    #                  [[1,tau*w],[-tau*w,1]]*x 
    osc = nengo.Ensemble(n_neurons=1000, dimensions=3,
                         radius=1.5)
    
    speed = nengo.Ensemble(n_neurons=100, dimensions=1)
    stim_speed = nengo.Node(0)
    nengo.Connection(stim_speed, speed)
    
    tau = 0.1
    w = 6
    def feedback(x):
        x0, x1, speed = x
        return x0+tau*w*speed*x1, -tau*w*speed*x0+x1
    nengo.Connection(osc, osc[0:2], function=feedback, 
                     synapse=tau)
    nengo.Connection(speed, osc[2])
    
    #def stim_func(t):
    #    if t < 0.1:
    #        return 1
    #    else:
    #        return 0
    #stim = nengo.Node(stim_func)
    stim = nengo.Node(lambda t: 1 if t<0.1 else 0)
    nengo.Connection(stim, osc[0])
    
    
        
        
        