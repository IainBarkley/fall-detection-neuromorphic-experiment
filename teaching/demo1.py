import nengo
import numpy as np

model = nengo.Network()
with model:
    my_neuron = nengo.Ensemble(n_neurons=2, 
                               dimensions=1,
                               encoders=[[1],[-1]],
                               max_rates=[100, 90],
                               intercepts=[-0.5, -0.5],
                               neuron_type=nengo.LIF(),
                               radius=np.pi,
                               )
                               
    stim = nengo.Node(1)
    #nengo.Connection(stim, my_neuron.neurons,
    #                    transform=[[0.5], [0.7]])
    nengo.Connection(stim, my_neuron)

                               
    