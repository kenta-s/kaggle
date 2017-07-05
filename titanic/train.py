from variable_builder import VariableBuilder
from chainer import Variable, optimizers, serializers
from IPython import embed
from titanic_chain import TitanicChain

variable_builder = VariableBuilder()
x_train_data = Variable(variable_builder())
# y_train_data
row, col = train_data.shape

titanic_chain = TitanicChain(col)

embed()
