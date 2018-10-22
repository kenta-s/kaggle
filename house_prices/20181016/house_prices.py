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

from datetime import datetime
from IPython import embed
import re
import glob
from data_augmentation import DataAugmentation
import xgboost as xgb
from xgboost import plot_importance

from lotfrontage_filler import LotfrontageFiller

def drop_columns(df):
    to_be_used = df.corr()[df.corr().SalePrice.abs() > 0.2].index
    
    # GarageYrBltはYearBltとの相関が高い 0.825667
    # 1stFlrSFはTotalBsmtSFとの相関が高い 0.819530
    # GarageAreaはGarageCarsとの相関が高い 0.882475
    # TotRmsAbvGrdはGrLivAreaとの相関が高い 0.828094
    to_be_used = to_be_used.drop(["GarageYrBlt", "1stFlrSF", "GarageArea", "TotRmsAbvGrd"])
    return df[to_be_used]
# da = DataAugmentation()
# da.augment()
def lotfrontage(df):
    df = df.dropna(subset=['MasVnrType', 'MasVnrArea'])
    df = pd.get_dummies(df)
    l = LotfrontageFiller()
    lotfrontage = l.predict(df)
    return lotfrontage

df = pd.read_csv("train.csv")
# MasVnrType, MasVnrAreaの欠損値は入力漏れっぽい？除外。8 rows
df = df.dropna(subset=['MasVnrType', 'MasVnrArea'])
lotfrontage = lotfrontage(df)
df.loc[df.LotFrontage.isnull(), 'LotFrontage'] = lotfrontage[df.LotFrontage.isnull()]
test_df = pd.read_csv("../test.csv")
all_df = pd.concat([df, test_df])

# これやると精度下がる？
# all_df = drop_columns(all_df)
original_df = df.copy()

train_num = len(df)
all_df = pd.get_dummies(all_df)
df = all_df[:train_num].copy()
test_df = all_df[train_num:].copy()

print('rows: {}'.format(len(df)))

X = df.drop('SalePrice', axis=1)
Y = df.SalePrice
x_train, x_test, y_train, y_test = sk.train_test_split(X, Y, test_size=0.01)
# clf = RandomForestRegressor(n_estimators=1000, max_depth=8, n_jobs=20)
clf = xgb.XGBRegressor()
clf.fit(x_train, y_train)

rmse = np.sqrt(mean_squared_error(y_test, clf.predict(x_test)))
print("RMSE: " + str(rmse))

raw_data = pd.read_csv("../test.csv")
test_data = test_df.drop('SalePrice', axis=1)
result = clf.predict(test_data)
files = glob.glob('csvs/*')
scores = list(map(lambda x: int(re.sub(r'\D', '', x)), files))
best_score = min(scores)
ans_csv = pd.concat((raw_data.Id, pd.DataFrame(result)), axis=1)
ans_csv.columns = ["Id", "SalePrice"]
ans_csv.to_csv('csvs/{rmse}_{time}.csv'.format(rmse=int(rmse), time=int(datetime.now().timestamp())), index=False)

embed()

# Id == 314 のSalePriceがやたら高く、これをうまくpredictできてなかったので調べる -> 削除した
# Id == 969 のSalePriceは低すぎて外してた