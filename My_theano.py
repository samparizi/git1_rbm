from __future__ import print_function

import timeit

try:
    import PIL.Image as Image
except ImportError:
    import Image

import numpy

import theano
import theano.tensor as T
import os

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import tile_raster_images
from logistic_sgd import load_data




import numpy as np


class RBM(object):
    def __init__(
            self,
            n_vis=784,
            n_hid=500,
            W=None,
            v_bias=None,
            h_bias=None,
            numpy_rng=None,
            theano_rng=None
    ):
        self.n_vis = n_vis
        self.n_hid = n_hid

    numpy_rng = np.random.RandomState(1234)
    theano_rng = RandomStreams
