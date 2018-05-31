
import os
import pandas as pd
from hyperopt import fmin, tpe, hp
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV



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


    model = lgb.LGBMClassifier(boosting_type='gbdt',objective='binary',
                              seed=123,
                              is_unbalance=True,
                              nthread=-1)


    param_dist = {'num_leaves': range(2, 31, 2),
                  'learning_rate': np.linspace(0.01,2,20),
                  'n_estimators': range(80,200,4),
                  'colsample_bytree': np.linspace(0.5,0.98,10),
                  'subsample': np.linspace(0.5,0.98,10),
                  'subsample_freq': range(1, 5),
                  'reg_alpha': np.linspace(0, 1, 10),
                  'reg_lambda': np.linspace(0, 1, 10),
                  }

    grid = RandomizedSearchCV(model, param_dist, cv=3, scoring='roc_auc',n_iter=50)

    #在训练集上训练
    grid.fit(xTrain,yTrain)
    #返回最优的训练器
    best_estimator = grid.best_estimator_
    print(best_estimator)
    #输出最优训练器的精度
    print(grid.best_score_)
