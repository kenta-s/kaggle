import numpy as np
# import matplotlib.pyplot as plt
# import chainer
# from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

class TitanicChain(Chain):
    def __init__(self, input_length):
        super(TitanicChain, self).__init__(
            l1 = L.Linear(input_length, 5),
            l2 = L.Linear(5, 3),
            l3 = L.Linear(3, 1)
        )

    def __call__(self, x, y):
        return F.mean_squared_error(self.fwd(x), y)
        # return F.binary_accuracy(self.fwd(x), y)

    def fwd(self, x):
        h1 = self.l1(x)
        h2 = self.l2(h1)
        h3 = self.l3(h2)
        return F.sigmoid(h3)
