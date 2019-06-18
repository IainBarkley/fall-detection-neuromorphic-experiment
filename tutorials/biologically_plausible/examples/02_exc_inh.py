import nengo
import nengo_bio as bio

with nengo.Network() as model:
    node_in = nengo.Node(0)
    ens_a = bio.Ensemble(n_neurons=50, dimensions=1, p_exc=0.8)
    ens_b = bio.Ensemble(n_neurons=50, dimensions=1)

    nengo.Connection(node_in, ens_a)
    bio.Connection(ens_a, ens_b)
