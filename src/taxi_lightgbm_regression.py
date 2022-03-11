from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import lightgbm as lgb

from azureml.core import Workspace, Datastore, Dataset, Run
from azureml.opendatasets import NycTlcGreen
import mlflow

import argparse
import time
import cloudpickle
import copy
from datetime import datetime
from dateutil.relativedelta import relativedelta

# 引数取得

parser = argparse.ArgumentParser()
parser.add_argument("--input-data", type=str)
parser.add_argument("--boosting_type", type=str, default='gbdt')
parser.add_argument("--metric", type=str, default='rmse')
parser.add_argument("--learning_rate", type=float, default=0.1)
parser.add_argument("--num_leaves", type=int, default=10)
parser.add_argument("--min_data_in_leaf", type=int, default=1)
parser.add_argument("--num_iteration", type=int, default=100)

args = parser.parse_args()

params = {
    'task': 'train',
    'boosting_type': args.boosting_type,
    'objective': 'regression',
    'metric': args.metric,
    'learning_rate': args.learning_rate,
    'num_leaves': args.num_leaves,
    'min_data_in_leaf': args.min_data_in_leaf,
    'num_iteration': args.num_iteration,
}

# mlflow autolog 開始
## ジョブ実行の場合 Azure ML が初期設定する環境変数をもとに mlflow が自動でセッティングされる
## 対話的な実験で実行したような URI の取得とセットは不要
## mlflow_uri = ws.get_mlflow_tracking_uri()
## mlflow.set_tracking_uri(mlflow_uri)

mlflow.lightgbm.autolog()

# パラメーター記録

mlflow.log_params(params)

# Workspace インスタンス取得

run = Run.get_context()
ws = run.experiment.workspace

# 入力した Dataset インスタンス取得

dataset = run.input_datasets["nyc_taxi_dataset"]

# データ取得＆加工

df = dataset.to_pandas_dataframe() 

train, test = train_test_split(df, test_size=0.2, random_state=1234)

x_train = train[train.columns[train.columns != 'totalAmount']]
y_train = train['totalAmount']

x_test = test[test.columns[test.columns != 'totalAmount']]
y_test = test['totalAmount']

lgb_train = lgb.Dataset(x_train, y_train)
lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)

# 学習

gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round=50,
    valid_sets=lgb_eval,
    early_stopping_rounds=10
)

# model保存

model_path = 'outputs/model.pickle'

with open(model_path, mode='wb') as f:
    cloudpickle.dump(gbm, f)

# テスト    

y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration)

# スコア取得＆記録

test_score = r2_score(y_test, y_pred)
print(test_score)
mlflow.log_metric('r2', test_score)

test_RMSE_score = np.sqrt(mean_squared_error(y_test, y_pred))
print(test_RMSE_score)
mlflow.log_metric('rmse', test_RMSE_score)