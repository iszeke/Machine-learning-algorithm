

import os
import pandas as pd
from hyperopt import fmin, tpe, hp
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import cross_val_score

def lgbt_optimize(argsDict):


    # 对参数做相应的转换
    num_leaves = int(argsDict['num_leaves'])
    learning_rate = float(argsDict["learning_rate"])
    n_estimators = int(argsDict["n_estimators"])
    colsample_bytree = float(argsDict["colsample_bytree"])
    subsample = float(argsDict["subsample"])
    subsample_freq = int(argsDict["subsample_freq"])
    reg_alpha = float(argsDict["reg_alpha"])
    reg_lambda = float(argsDict["reg_lambda"])

    # 打印参数字典
    print(argsDict)
    # 设为全局变量
    global xTrain, yTrain

    model = lgb.LGBMClassifier(boosting_type='gbdt',
                               objective='binary',
                               num_leaves=num_leaves, # 叶子数
                               learning_rate=learning_rate, #学习率
                               colsample_bytree=colsample_bytree,
                               subsample=subsample, #采样数
                               subsample_freq=subsample_freq,
                               reg_alpha=reg_alpha,
                               reg_lambda=reg_lambda,
                               n_estimators=n_estimators, #树的数量
                               is_unbalance=True,
                               seed=123,
                               nthread=-1)

    model.fit(xTrain, yTrain)

    yTest_pred_proba = pd.DataFrame(model.predict_proba(xTest)).iloc[:, 1]
    yTest_pred_label = yTest_pred_proba.apply(lambda x: 0 if x<=0.5 else 1)

    metric = cross_val_score(model,xTrain, yTrain,cv=5,scoring="roc_auc").mean()
    print(metric)
    return -metric



if __name__ == '__main__':

    titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
    y = titanic['survived']
    X = titanic.drop(['row.names','name', 'survived'],axis=1)

    # 填充空值
    X['age'].fillna(X['age'].mean(), inplace=True)
    X.fillna('UNKNOWN', inplace=True)

    # 分割数据
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.25, random_state=123)

    # 类别型特征向量化
    from sklearn.feature_extraction import DictVectorizer
    vec = DictVectorizer(sparse=False)
    xTrain = vec.fit_transform(xTrain.to_dict(orient='records'))
    xTest = vec.transform((xTest.to_dict(orient='records')))
    print(len(vec.feature_names_))

      

    space = {'num_leaves': hp.uniform('num_leaves', 2, 31),
             'learning_rate': hp.uniform('learning_rate', 0.01, 0.1),
             'n_estimators': hp.randint('n_estimators', 1000),
             'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 1),
             'subsample': hp.uniform('subsample', 0.1, 1),
             'subsample_freq': hp.uniform('subsample_freq', 1, 5),
             'reg_alpha': hp.uniform('reg_alpha', 0, 1),
             'reg_lambda': hp.uniform('reg_lambda', 0, 1)
             }

    best_para = fmin(lgbt_optimize, space, algo=tpe.suggest, max_evals=50)

        
    print('')
    print('best_para', best_para)

    print(lgbt_optimize(best_para))























