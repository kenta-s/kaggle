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
        self.doc.load_from_text('house_price.dict')
        for row in self.df.values.tolist():
            strs = list(filter(lambda x: type(x) == str, row))
            self.doc.add_documents([strs])
        self.doc.save_as_text('house_price.dict')

    def build_np_array(self):
        tmp = self.df.drop("Alley", axis=1)
        tmp = tmp.drop("PoolQC", axis=1)
        tmp = tmp.drop("MiscFeature", axis=1)
        tmp = tmp.drop("Fence", axis=1)

        tmp = tmp.drop("SalePrice", axis=1)
        for row in tmp.values.tolist():
            array = list(map(self.convert_to_float, row))
            self.train_data.append(np.array(array).astype(np.float32))

        return self.train_data

    def build_y(self):
        train = []
        for i in range(len(self.train_data)):
            train.append((self.train_data[i], np.array([self.df.SalePrice[i]], dtype=np.float32)))

        return train

    def calc_avg(self, col):
        col_dict = {}
        target_col_list = list(set(self.df[col]))
        for c in target_col_list:
            col_dict[str(c)] = {"count":0, "total":0}
        for row in self.df[[col, "SalePrice"]].values.tolist():
            col_dict[str(row[0])]["count"] += 1
            col_dict[str(row[0])]["total"] += row[1]

        for c in target_col_list :
            col_dict[str(c)]["avg"] = col_dict[str(c)]["total"] / col_dict[str(c)]["count"]
            
        return col_dict
       

    def build_np_array_for_submission(self):
        for row in self.df.values.tolist():
            array = list(map(self.convert_to_float, row))
            self.train_data.append(np.array(array).astype(np.float32))

        return self.train_data

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
