import pandas as pd
import numpy as np
from chainer import Variable, optimizers, serializers
from variable_builder import VariableBuilder
from titanic_chain import TitanicChain
from IPython import embed

variable_builder = VariableBuilder('test.csv')
x_test = variable_builder.build_variable_x()
row, col = x_test.shape

model = TitanicChain(col)
serializers.load_npz('titanic.npz', model)

answer = model.fwd(x_test)
def titanic_round(n):
    if n >= 0.5:
        return 1
    else:
        return 0

dead_or_alive_list = list(map(titanic_round, answer.data))

passenger_id_list = list(pd.read_csv('test.csv').PassengerId)
answer = {'PassengerId': passenger_id_list, 'Survived': dead_or_alive_list}
df = pd.DataFrame(answer)
df.to_csv('answer.csv', index=None)
