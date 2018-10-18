from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import sklearn.model_selection as sk
from gensim import corpora
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from IPython import embed

df = pd.read_csv("train.csv")
processed_df = df
nans = (df.isnull().sum() / len(df)) * 100
for colum in list(df):
    if(nans[colum] != 0):
        print(colum)
        print(df[colum].isnull().sum())

transform_list = ["MSZoning", "Street"]
for column in transform_list:
    le = LabelEncoder()
    le.fit(df[column])
    processed_df[column] = le.transform(df[column])

fillna_columns = ["LotFrontage"]
for column in fillna_columns:
    processed_df[column] = processed_df[column].fillna(0)

# sns.pairplot(df[['Street', 'SalePrice']], hue="Street")
embed()