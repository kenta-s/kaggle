from data_builder import DataBuilder
from chainer import Variable, optimizers, serializers, iterators, training
import chainer.links as L
from chainer.training import extensions
import numpy as np
from IPython import embed
from house_price_chain import HousePriceChain, Classifier

data_builder = DataBuilder('train.csv')
whole_train_data = data_builder()
train = whole_train_data[0:int(len(whole_train_data) / 2)]
test = whole_train_data[int(len(whole_train_data) / 2):int(len(whole_train_data))]

col = len(train[0])

train_iter = iterators.SerialIterator(train, batch_size=20, shuffle=True)
test_iter = iterators.SerialIterator(test, batch_size=20, repeat=False, shuffle=False)

model = Classifier(HousePriceChain(col, 1))
optimizer = optimizers.Adam()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (1000, 'epoch'), out='result')
trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss']))
trainer.extend(extensions.ProgressBar())
trainer.run()
