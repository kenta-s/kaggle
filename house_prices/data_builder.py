import numpy as np
import pandas as pd
from gensim import corpora, matutils
from IPython import embed

class DataBuilder():
    def __init__(self, file):
        self.df = pd.read_csv(file)
        self.doc = corpora.Dictionary([[]])
        self.train_data = []

    def __call__(self):
        self.build_doc()
        return self.build_np_array()

    def build_doc(self):
        for row in self.df.values.tolist():
            strs = list(filter(lambda x: type(x) == str, row))
            self.doc.add_documents([strs])

    def build_np_array(self):
        for row in self.df.drop("SalePrice", axis=1).values.tolist():
            array = list(map(lambda x: self.doc.token2id[x] if type(x) == str else x, row))
            self.train_data.append(np.array(array).astype(np.float32))

        return self.train_data

d = DataBuilder('train.csv')
# d.build_doc()
# d.build_np_array()
embed()
