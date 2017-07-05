from variable_builder import VariableBuilder
from chainer import Variable, optimizers, serializers
from IPython import embed
from titanic_chain import TitanicChain

variable_builder = VariableBuilder()
x_train = Variable(variable_builder.build_variable_x())
y_train = Variable(variable_builder.build_variable_y())
row, col = x_train.shape

titanic_chain = TitanicChain(col)

embed()
