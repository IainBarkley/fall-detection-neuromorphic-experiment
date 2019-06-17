import nengo
import numpy as np
from shapes import circle, rectangle, triangle, image, canvas


class Environment(object):

    def __init__(self, img1, img2, size=10):

        self.size = size
        self.img1 = img1
        self.img2 = img2

        self._nengo_html_ = ''

    def __call__(self, t, x):

        seed = int(x[0])
        n_images = max(1, int(x[1]))

        rstate = np.random.RandomState(seed=seed)

        locations = rstate.uniform(low=0, high=self.size, size=(n_images, 2))

        shape_list = []

        for i in range(n_images):
            # only have the first random image be different
            if i == 0:
                img = self.img1
            else:
                img = self.img2
            shape_list.append(
                image(height=1, width=1, x=locations[i, 0], y=locations[i, 1], img=img)
            )
        
        # draw all of the images on the screen
        self._nengo_html_ = canvas(shape_list, width=self.size, height=self.size)

img1 = 'images/icons8-badger-96.png'
img2 = 'images/icons8-fox-96.png'


model = nengo.Network(seed=13)

with model:

    seed_slider = nengo.Node([0])
    number_of_images = nengo.Node([0])

    env = nengo.Node(
        Environment(
            img1=img1,
            img2=img2,
            size=10,
        ),
        size_in=2,
        size_out=0,
    )

    nengo.Connection(seed_slider, env[0])
    nengo.Connection(number_of_images, env[1])

