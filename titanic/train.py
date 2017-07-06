from variable_builder import VariableBuilder
from chainer import Variable, optimizers, serializers
import numpy as np
from IPython import embed
from titanic_chain import TitanicChain

variable_builder = VariableBuilder()
x_train = variable_builder.build_variable_x()
y_train = variable_builder.build_variable_y()
row, col = x_train.shape

model = TitanicChain(col)
optimizer = optimizers.Adam()
optimizer.setup(model)

n = len(x_train)
bs = 25
for j in range(1000):
    sffindx = np.random.permutation(n)
    for i in range(0, n, bs):
        idx = sffindx[i:(i+bs) if (i+bs) < n else n]
        x = Variable(x_train[idx])
        y = Variable(y_train[idx])
        model.zerograds()
        loss = model(x, y)
        loss.backward()
        optimizer.update()

embed()
