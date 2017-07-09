import numpy as np
# import matplotlib.pyplot as plt
# import chainer
# from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

# class TitanicChain(Chain):
#     def __init__(self, input_length):
#         super(TitanicChain, self).__init__(
#             l1 = L.Linear(input_length, 6),
#             l2 = L.Linear(6, 3),
#             l3 = L.Linear(3, 1)
#         )
#
#     def __call__(self, x):
#         return F.mean_squared_error(self.fwd(x), y)
#
#     def fwd(self, x):
#         h1 = self.l1(x)
#         h2 = self.l2(h1)
#         h3 = self.l3(h2)
#         return F.sigmoid(h3)

class TitanicChain(Chain):
    def __init__(self, n_units, n_out):
        super(TitanicChain, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)    # n_units -> n_out

    def __call__(self, x):
        h1 = self.l1(x)
        h2 = self.l2(h1)
        h3 = self.l3(h2)
        return F.sigmoid(h3)
