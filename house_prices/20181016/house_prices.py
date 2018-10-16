from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import sklearn.model_selection as sk
from gensim import corpora
import seaborn as sns
import matplotlib.pyplot as plt

from IPython import embed

df = pd.read_csv("train.csv")
nans = (df.isnull().sum() / len(df)) * 100
for colum in list(df):
    if(nans[colum] != 0):
        print(colum)
        print(df[colum].isnull().sum())
df.LotFrontage.fillna(0)
embed()