from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import sklearn.model_selection as sk
from gensim import corpora
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

from IPython import embed

df = pd.read_csv("train.csv")
to_be_used = df.corr()[df.corr().SalePrice > 0.2].index

# GarageYrBltはYearBltとの相関が高い 0.825667
to_be_used = to_be_used.drop(["GarageYrBlt"])
processed_df = df[to_be_used]

def process_data(target_df):
    for column in target_df.columns:
        target_df[column] = target_df[column].fillna(0)
    return target_df

processed_df = process_data(processed_df)

X = processed_df.drop('SalePrice', axis=1)
Y = processed_df.SalePrice
x_train, x_test, y_train, y_test = sk.train_test_split(X, Y, test_size=0.1)
clf = RandomForestRegressor(n_estimators=1300, max_depth=7, n_jobs=10)
clf.fit(x_train, y_train)

rmse = np.sqrt(mean_squared_error(y_test, clf.predict(x_test)))
print("RMSE: " + str(rmse))

raw_data = pd.read_csv("../test.csv")
test_data = pd.read_csv("../test.csv")[to_be_used.drop('SalePrice')]
test_data = process_data(test_data)
result = clf.predict(test_data)
ans_csv = pd.concat((raw_data.Id, pd.DataFrame(result)), axis=1)
ans_csv.columns = ["Id", "SalePrice"]
ans_csv.to_csv('csvs/{rmse}.csv'.format(rmse=int(rmse)), index=False)