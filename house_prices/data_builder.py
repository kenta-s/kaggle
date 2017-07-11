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
        self.build_np_array()
        return self.build_y()

    def build_doc(self):
        for row in self.df.values.tolist():
            strs = list(filter(lambda x: type(x) == str, row))
            self.doc.add_documents([strs])

    def build_np_array(self):
        for row in self.df.drop("SalePrice", axis=1).values.tolist():
            array = list(map(self.convert_to_float, row))
            self.train_data.append(np.array(array).astype(np.float32))

        return self.train_data

    def build_y(self):
        train = []
        for i in range(len(self.train_data)):
            train.append((self.train_data[i], np.array([self.df.SalePrice[i]], dtype=np.float32)))

        return train

    def convert_to_float(self, x):
        if type(x) == str:
            flt = self.doc.token2id[x]
            if np.isnan(flt):
                return 0.0
            else:
                return flt
        elif np.isnan(x):
            return 0.0
        else:
            return x

# d = DataBuilder('train.csv')
# d.build_doc()
# d.build_np_array()
