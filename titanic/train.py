from variable_builder import VariableBuilder
from chainer import Variable, optimizers, serializers, iterators, training
import chainer.links as L
from chainer.training import extensions
import numpy as np
from IPython import embed
from titanic_chain import TitanicChain

variable_builder = VariableBuilder('train.csv')
train = variable_builder.build_train_variable()

variable_builder2 = VariableBuilder('test.csv')
test = variable_builder2.build_test_variable('gender_submission.csv')

col = len(train[0])

train_iter = iterators.SerialIterator(train, batch_size=100, shuffle=True)
test_iter = iterators.SerialIterator(test, batch_size=100, repeat=False, shuffle=False)

model = L.Classifier(TitanicChain(col, 2))
optimizer = optimizers.Adam()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (1000, 'epoch'), out='result')
trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())
trainer.run()
