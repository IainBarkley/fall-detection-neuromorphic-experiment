import nengo
import numpy as np

model = nengo.Network()
with model:
    food = nengo.Ensemble(n_neurons=200, dimensions=2)
    stim_food = nengo.Node([0,0])
    nengo.Connection(stim_food, food)
    
    safe = nengo.Ensemble(n_neurons=100, dimensions=1)
    stim_safe = nengo.Node([0])
    nengo.Connection(stim_safe, safe)
    
    motor = nengo.Ensemble(n_neurons=200, dimensions=2)
    
    do_food = nengo.Ensemble(n_neurons=300, dimensions=3)
    nengo.Connection(food, do_food[:2])
    nengo.Connection(safe, do_food[2])
    
    def do_food_func(x):
        food_x, food_y, safe = x
        if safe>0:
            norm = np.linalg.norm([food_x, food_y])
            if norm > 0.001:
                return food_x/norm, food_y/norm
        return 0,0
    nengo.Connection(do_food, motor, function=do_food_func)
    
    position = nengo.Ensemble(n_neurons=500, dimensions=2)
    nengo.Connection(position, position, synapse=0.1)
    #def scale_func(x):
    #    return x*0.1
    #nengo.Connection(motor, position, function=scale_func)
    nengo.Connection(motor, position, transform=0.1, synapse=0.1)
    
    do_home = nengo.Ensemble(n_neurons=300, dimensions=3)
    nengo.Connection(position, do_home[:2])
    nengo.Connection(safe, do_home[2])
    
    def do_home_func(x):
        pos_x, pos_y, safe = x
        if safe<0:
            norm = np.linalg.norm([pos_x, pos_y])
            if norm > 0.001:
                return -pos_x/norm, -pos_y/norm
        return 0,0
    nengo.Connection(do_home, motor, function=do_home_func)
    
    