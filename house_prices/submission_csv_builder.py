from data_builder import DataBuilder
from chainer import Variable, optimizers, serializers, iterators, training
import chainer.links as L
from chainer.training import extensions
import numpy as np
from IPython import embed
from house_price_chain import HousePriceChain, Classifier

data_builder = DataBuilder('test.csv')
data_builder.build_doc()
test = data_builder.build_np_array_for_submission()

col = len(test[0])

model = Classifier(HousePriceChain(col, 1))
optimizer = optimizers.Adam()
optimizer.setup(model)

serializers.load_npz('house_price.npz', model)

embed()
