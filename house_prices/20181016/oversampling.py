from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython import embed

class Oversampling():
    def __init__(self, df):
        self.df = df
        self.sm = SMOTE(ratio='auto', k_neighbors=5)
    
    def exe(self):
        # TODO: SalePrice以外も追加
        return self.sm.fit_sample(self.df[['SalePrice']], (self.df.SalePrice > 240000))

df = pd.read_csv("train.csv")
o = Oversampling(df)
o.exe()