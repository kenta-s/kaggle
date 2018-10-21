import pandas as pd
import numpy as np
from IPython import embed

pd.set_option('display.max_columns', 500)

class DataAugmentation():
    def __init__(self):
        self.df = pd.read_csv("train.csv")
        self.original_df = self.df.copy()
    
    def augment(self):
        columns = self.target_columns()
        df = self.df.query("SalePrice > 210000")
        df = df.dropna(subset=['MasVnrType', 'MasVnrArea', 'LotFrontage'])
        new_df = df.copy()
        new_df2 = df.copy()
        for column in columns:
            mul = np.random.randint(1, 4, (len(df))) * 0.01
            new_df[column] = df[column] + (df[column] * mul)
        for column in columns:
            mul = np.random.randint(1, 4, (len(df))) * 0.01
            new_df2[column] = df[column] - (df[column] * mul)
        df = self.original_df.append(new_df)
        df = df.append(new_df2)
        df.to_csv('augmented.csv')
    
    def target_columns(self):
        columns = [
            'LotFrontage',
            'PoolArea',
            'GarageArea',
            "WoodDeckSF",
            "LotArea",
            "OpenPorchSF",
            'GrLivArea',
            '1stFlrSF',
            '2ndFlrSF',
            'TotalBsmtSF',
            'BsmtUnfSF',
            'BsmtFinSF1',
            'MasVnrArea',
            "SalePrice"
        ]
        return columns
        # return [column for column in self.df.columns.drop('Id') if len(self.df[column].unique()) > 300]