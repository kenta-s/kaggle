from variable_builder import VariableBuilder
from chainer import Variable, optimizers, serializers
import numpy as np
from IPython import embed
from titanic_chain import TitanicChain

variable_builder = VariableBuilder('train.csv')
x_train = variable_builder.build_variable_x()
y_train = variable_builder.build_variable_y()
row, col = x_train.shape

model = TitanicChain(col)
optimizer = optimizers.Adam()
optimizer.setup(model)

epoch = 2000
n = len(x_train)
bs = 25
for j in range(epoch):
    sffindx = np.random.permutation(n)
    for i in range(0, n, bs):
        idx = sffindx[i:(i+bs) if (i+bs) < n else n]
        x = Variable(x_train[idx])
        y = Variable(y_train[idx])
        model.zerograds()
        loss = model(x, y)
        loss.backward()
        optimizer.update()

    print('epoch:' + str(j + 1))

variable_builder2 = VariableBuilder('test.csv')
x_test = variable_builder2.build_variable_x()

variable_builder3 = VariableBuilder('gender_submission.csv')
y_test = variable_builder3.build_variable_y()

y = model.fwd(x_test)
alive = y.data >= 0.5

new_y = np.zeros(len(y)).reshape(len(y),1)
new_y[alive] = 1
ok = 0
for i in range(len(y)):
    if new_y[i] == y_test[i]:
        ok += 1

percent = ok / len(y)
print(percent)

if percent > 0.96:
  serializers.save_npz('titanic.npz', model)
