import numpy as np
import nengo
import redis


class Receiver(object):

    def __init__(self):

        self.r = redis.StrictRedis(host='localhost')

    def __call__(self, t):
        
        return float(self.r.get('0'))

model = nengo.Network()
with model:
    ens = nengo.Ensemble(n_neurons=100, dimensions=1)
    receiver_node = nengo.Node(
        Receiver(),
        size_in=0,
        size_out=1
    )

    nengo.Connection(receiver_node, ens)

