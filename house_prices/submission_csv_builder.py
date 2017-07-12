from data_builder import DataBuilder
from chainer import Variable, optimizers, serializers, iterators, training
import chainer.links as L
from chainer.training import extensions
import numpy as np
from IPython import embed
from house_price_chain import HousePriceChain, Classifier
import pandas as pd

data_builder = DataBuilder('test.csv')
data_builder.build_doc()
test = data_builder.build_np_array_for_submission()

col = len(test[0])

model = Classifier(HousePriceChain(col, 1))
optimizer = optimizers.Adam()
optimizer.setup(model)

serializers.load_npz('house_price.npz', model)

def predict(target):
    return int(model.predictor(np.array([target])).data)

house_price_ids = list(pd.read_csv('test.csv').Id)
house_price_list = list(map(predict, test))
answer = {'Id': house_price_ids, 'SalePrice': house_price_list}
df = pd.DataFrame(answer)
df.to_csv('answer.csv', index=None)
