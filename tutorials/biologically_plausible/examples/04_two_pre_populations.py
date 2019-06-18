import nengo
import nengo_bio as bio
import numpy as np

with nengo.Network() as model:
    node_in = nengo.Node((0, 0))
    ens_a = bio.Ensemble(n_neurons=50, dimensions=1, p_exc=0.8)
    ens_b = bio.Ensemble(n_neurons=50, dimensions=1, p_inh=0.2)
#    ens_a = bio.Ensemble(n_neurons=50, dimensions=1, p_exc=1.0)
#    ens_b = bio.Ensemble(n_neurons=50, dimensions=1, p_inh=1.0)
    ens_c = bio.Ensemble(n_neurons=50, dimensions=1)

    nengo.Connection(node_in[0], ens_a)
    nengo.Connection(node_in[1], ens_b)
    bio.Connection((ens_a, ens_b), ens_c, function=lambda x: np.mean(x))
