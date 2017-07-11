import numpy as np
from chainer import Link, Chain, ChainList
from chainer import report
import chainer.functions as F
import chainer.links as L

class Classifier(Chain):
    def __init__(self, predictor):
        super(Classifier, self).__init__()
        with self.init_scope():
            self.predictor = predictor

    def __call__(self, x, t):
        y = self.predictor(x)
        loss = F.mean_squared_error(y, t)
        report({'loss': loss}, self)
        return loss

class HousePriceChain(Chain):
    def __init__(self, n_units, n_out):
        super(HousePriceChain, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_units)
            self.l2 = L.Linear(None, n_units)
            self.l3 = L.Linear(None, n_out)

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = self.l3(h2)
        return h3
