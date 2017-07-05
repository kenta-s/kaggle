from variable_builder import VariableBuilder
from chainer import Variable, optimizers, serializers
from IPython import embed

variable_builder = VariableBuilder()
train_data = Variable(variable_builder())
