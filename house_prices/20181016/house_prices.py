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
import re
import glob

df = pd.read_csv("train.csv")

missingColumns = [
    'LotFrontage'
]

strColumns = [
    'MSZoning',
    'Street',
    'Alley',
    'LotShape',
    'LandContour',
    'Utilities',
    'Fence',
    'MiscFeature',
    'SaleType',
    'SaleCondition',
    'PoolQC',
    'BldgType',
    'BsmtCond',
    'BsmtExposure',
    'BsmtFinType1',
    'BsmtFinType2',
    'BsmtQual',
    'CentralAir',
    'Condition1',
    'Condition2',
    'Electrical',
    'ExterCond',
    'ExterQual',
    'Exterior1st',
    'Exterior2nd',
    'FireplaceQu',
    'Foundation',
    'Functional',
    'GarageCond',
    'GarageFinish',
    'GarageQual',
    'GarageType',
    'Heating',
    'HeatingQC',
    'HouseStyle',
    'KitchenQual',
    'LandSlope',
    'LotConfig',
    'MasVnrType',
    'Neighborhood',
    'PavedDrive',
    'RoofMatl',
    'RoofStyle'
]

processed_df = df
def missing(target_df):
    for column in missingColumns:
        new_column = 'missing_{}'.format(column)
        index = target_df[column].isnull()
        target_df[new_column] = index
    return target_df

def transform_df(target_df):
    for column in strColumns:
        target_df[column] = target_df[column].fillna('None')
        le = LabelEncoder()
        le.fit(target_df[column].unique())
        target_df[column] = le.transform(target_df[column])
    return target_df

def process_data(target_df):
    for column in target_df.columns:
        if str in map(lambda x: type(x), target_df[column]):
            target_df[column] = target_df[column].fillna('None')
        elif column in missingColumns:
            target_df = missing(target_df)
            target_df[column] = target_df[column].fillna(target_df[column].mean())
        else:
            target_df[column] = target_df[column].fillna(0)
    return target_df

processed_df = transform_df(processed_df)
processed_df = process_data(processed_df)

to_be_used = processed_df.corr()[processed_df.corr().SalePrice.abs() > 0.2].index

# GarageYrBltはYearBltとの相関が高い 0.825667
# 1stFlrSFはTotalBsmtSFとの相関が高い 0.819530
# GarageAreaはGarageCarsとの相関が高い 0.882475
to_be_used = to_be_used.drop(["GarageYrBlt", "1stFlrSF", "GarageArea"])
processed_df = df[to_be_used]

X = processed_df.drop('SalePrice', axis=1)
Y = processed_df.SalePrice
x_train, x_test, y_train, y_test = sk.train_test_split(X, Y, test_size=0.1)
clf = RandomForestRegressor(n_estimators=2400, max_depth=11, n_jobs=20)
clf.fit(x_train, y_train)

rmse = np.sqrt(mean_squared_error(y_test, clf.predict(x_test)))
print("RMSE: " + str(rmse))

raw_data = pd.read_csv("../test.csv")
test_data = transform_df(raw_data)
test_data = process_data(test_data)
test_data = test_data[to_be_used.drop('SalePrice')]
result = clf.predict(test_data)
files = glob.glob('csvs/*')
scores = list(map(lambda x: int(re.sub(r'\D', '', x)), files))
best_score = min(scores)
if rmse < best_score:
    ans_csv = pd.concat((raw_data.Id, pd.DataFrame(result)), axis=1)
    ans_csv.columns = ["Id", "SalePrice"]
    ans_csv.to_csv('csvs/{rmse}.csv'.format(rmse=int(rmse)), index=False)