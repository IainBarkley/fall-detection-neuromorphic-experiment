import nengo
import numpy as np

model = nengo.Network(seed=3)
with model:
    my_ensemble = nengo.Ensemble(n_neurons=100, 
                                 neuron_type=nengo.LIF(),
                                 dimensions=1,
                                 )
                                 
    stim = nengo.Node(0)
    
    nengo.Connection(stim, my_ensemble,
                     synapse=None)
                                 
    ensemble_2 = nengo.Ensemble(n_neurons=50, dimensions=1)
    
    nengo.Connection(my_ensemble, ensemble_2)
    
    # this is a NEF-style connection
    #nengo.Connection(ensemble_2, ensemble_2)
    
    # this is a traditional neural network connection 
    nengo.Connection(ensemble_2.neurons, ensemble_2.neurons, 
                     transform=np.ones((ensemble_2.n_neurons,
                                        ensemble_2.n_neurons))*(-0.01))
    
    #nengo.Connection(ensemble_2.neurons[0:10], ensemble_2.neurons[10:20], 
    #                 transform=np.ones((10,
    #                                    10))*(-0.01))
    
    