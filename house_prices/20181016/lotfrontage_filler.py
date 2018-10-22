import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import sklearn.model_selection as sk
from IPython import embed

class LotfrontageFiller():
    def __init__(self, df):
        self.df = self.process(df)
        self.x_train, self.x_test, self.y_train, self.y_test = sk.train_test_split(self.df.drop('LotFrontage', axis=1), self.df.LotFrontage, test_size=0.1)
        self.clf = RandomForestRegressor(n_estimators=30)
        self.clf.fit(self.x_train, self.y_train)
    
    def predict(self, df):
        return self.clf.predict(df)

    def process(self, df):
        df = df[df.LotFrontage.notnull()]
        df = df.drop(['GarageYrBlt','SalePrice'], axis=1)
        df = df.dropna(subset=['MasVnrType', 'MasVnrArea'])
        df = pd.get_dummies(df)
        return df

df = pd.read_csv("train.csv")
l = LotfrontageFiller(df)
embed()