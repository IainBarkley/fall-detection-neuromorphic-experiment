import nengo
import nengo_bio as bio
import numpy as np

with nengo.Network() as model:
    node_in = nengo.Node((0,))
    ens_a = bio.Ensemble(n_neurons=50, dimensions=1, p_exc=1.0)
    ens_b = bio.Ensemble(n_neurons=50, dimensions=1, p_inh=1.0)
    ens_c = bio.Ensemble(n_neurons=50, dimensions=1)

    nengo.Connection(node_in, ens_a)
    bio.Connection(ens_a, ens_b)
    bio.Connection((ens_a, ens_b), ens_c, function=lambda x: np.mean(x))
#    bio.Connection((ens_a, ens_b), ens_c, function=lambda x: np.mean(x)**2)
#    bio.Connection((ens_a, ens_b), ens_c,
#                   function=lambda x: np.mean(x)**2,
#                   eval_points=(np.linspace(-1, 1, 750)[:, None] * np.ones(2)))
#    bio.Connection({ens_a, ens_b}, ens_c, function=lambda x: x**2)
#    bio.Connection({ens_a, ens_b}, ens_c, function=lambda x: x**2,
#                   solver=bio.solvers.QPSolver(relax=True))

