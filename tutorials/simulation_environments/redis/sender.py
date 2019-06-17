import numpy as np
import nengo
import redis


class Sender(object):

    def __init__(self):

        self.r = redis.StrictRedis(host='localhost')

    def __call__(self, t, x):
        
        for i, data in enumerate(x):
            self.r.set(str(i), data)


model = nengo.Network()
with model:
    input_node = nengo.Node([0])
    sender_node = nengo.Node(
        Sender(),
        size_in=1,
        size_out=0
    )

    nengo.Connection(input_node, sender_node)

