

import os
import pandas as pd
from hyperopt import fmin, tpe, hp
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import time
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def data_process(file_name, col_name, ds, dt):
    # "对原始数据进行删除/填充/切割/变化等"

    with open(file_name, 'r') as f: # 读取文件进入dataframe
        df = pd.read_csv(f,header=0,index_col=0,low_memory=False)

    #防止文件中有空格，将空格替换为NaN,并且将表格转换成数值
    df = df.replace(' ', np.NaN)
    df = df.apply(pd.to_numeric, errors='ignore')
    
    #切片选择的列,并且以5min填充行
    df = df.loc[:, col_name]

    #切片训练时间
    df.index = pd.to_datetime(df.index)
    df = df.resample('10min',label='left').first()
    df = df.loc[ds:dt, :]

    #删除空值
    df = df.dropna(axis=0, how='any') #删除表中含有NaN的行
    #data = data.fillna(df.median(),inplace=True) #将空值填充为每一列的中值

    #筛选功率大于5的行，筛选掉停机工况
    df['r18'] = df['r18'].apply(float)
    df = df[df['r18'] > 5]

    return df

def lgbt_optimize(argsDict):

    # 对参数做相应的转换
    num_leaves = int(argsDict['num_leaves'])
    learning_rate = float(argsDict['learning_rate'])
    n_estimators = int(argsDict["n_estimators"])
    subsample = float(argsDict["subsample"])
    subsample_freq = int(argsDict["subsample_freq"])
    colsample_bytree = float(argsDict["colsample_bytree"])
    reg_alpha = float(argsDict["reg_alpha"])
    reg_lambda = float(argsDict["reg_lambda"])

    # 打印参数字典
    print(argsDict)

    # 声明全局变量
    global df_X, df_y
    X = df_X.values
    y = df_y.values.reshape(-1,1)
    model = lgb.LGBMRegressor(boosting_type='gbdt',
                              num_leaves=num_leaves,
                              learning_rate=learning_rate,
                              n_estimators=n_estimators,
                              max_bin=255,
                              subsample_for_bin=200000,
                              objective='regression',
                              min_split_gain=0.0,
                              min_child_weight=0.001,
                              min_child_samples=20, #叶子结点最小的样本数
                              subsample=subsample,
                              subsample_freq=subsample_freq,
                              colsample_bytree=colsample_bytree,
                              reg_alpha=reg_alpha,
                              reg_lambda=reg_lambda,
                              random_state=123,
                              n_jobs=-1)
    model.fit(X, y)
    # 评估
    metric = cross_val_score(model,X, y,cv=5,scoring="r2").mean()
    print(metric)
    # 保存metric结果到csv
    temp = pd.Series(argsDict)
    temp['metric'] = metric
    temp.to_csv(str(time.time()) + '.csv')

    return -metric

def opt_fig():
    fns = [x for x in os.listdir('.') if os.path.isfile(x) and os.path.splitext(x)[1] == '.csv']
    temp = True
    for fn in fns:
        if temp:
            pf = pd.read_csv(fn, index_col=0, header=None)
            temp = False
        else:
            pf1 = pd.read_csv(fn, index_col=0, header=None)
            pf = pd.concat([pf, pf1], axis=1)
    # print(pf.loc['metric', :])
    plt.scatter(np.arange(1, len(pf.loc['metric', :])+1), pf.loc['metric', :].values)
    plt.show()

def fu_pred(argsDict, df_X, df_y, df_X_fu, df_y_fu):

    # 对参数做相应的转换
    num_leaves = int(argsDict['num_leaves'])
    learning_rate = float(argsDict['learning_rate'])
    n_estimators = int(argsDict["n_estimators"])
    subsample = float(argsDict["subsample"])
    subsample_freq = int(argsDict["subsample_freq"])
    colsample_bytree = float(argsDict["colsample_bytree"])
    reg_alpha = float(argsDict["reg_alpha"])
    reg_lambda = float(argsDict["reg_lambda"])

    # 打印参数字典
    print(argsDict)

    X = df_X.values
    y = df_y.values.reshape(-1,1)

    model = lgb.LGBMRegressor(boosting_type='gbdt',
                              num_leaves=num_leaves,
                              learning_rate=learning_rate,
                              n_estimators=n_estimators,
                              max_bin=255,
                              subsample_for_bin=200000,
                              objective='regression',
                              min_split_gain=0.0,
                              min_child_weight=0.001,
                              min_child_samples=20, #叶子结点最小的样本数
                              subsample=subsample,
                              subsample_freq=subsample_freq,
                              colsample_bytree=colsample_bytree,
                              reg_alpha=reg_alpha,
                              reg_lambda=reg_lambda,
                              random_state=123,
                              n_jobs=-1)
    model.fit(X, y)
    # 预测
    y_fu_pred = model.predict(df_X_fu.values)

    print('fu r2_score:', r2_score(df_y_fu.values, y_fu_pred))
    plt.plot(df_y_fu.index, y_fu_pred)
    plt.plot(df_y_fu.index, df_y_fu.values)
    plt.show()





if __name__ == '__main__':

#1、需要填写的参数
##########################################################################

    #定义结果文件存储的路径
    os.chdir(r"C:\Users\Chinawindey\Desktop\test")

    #定义读取原始文件的路径
    file_name = r'D:\00_工作日志\S\齿轮箱状态监测\2018-03\sys_data\ZYBL_wt8_5min_2016-01_to_2018-01.csv'

    #定义构造记忆矩阵的输入时间
    ds = '2017-01-01'
    dt = '2017-05-31'

    #定义预测的时间
    ds_fu = '2016-01-01'
    dt_fu = '2016-02-01'

    #X变量
    col_name = ['r43','r47','r76','r78','r80','r18']

#2、选择最佳训练参数
#############################################################################   
    print('程序开始运行...')
    df = data_process(file_name,col_name,ds,dt)

    df_X = df.iloc[:, :-1]
    df_y = df.iloc[:, -1]

    # # 分割数据
    # xTrain, xTest, yTrain, yTest = train_test_split(df_X, df_y, test_size=0.2, random_state=123)
    # print(xTrain.head())
    # print(yTrain.head())
    #
    # #缩放x到[0,1]
    # mms = preprocessing.MinMaxScaler()
    #
    # xTrain = pd.DataFrame(data=mms.fit_transform(xTrain),index=xTrain.index,columns=xTrain.columns)
    # print(xTrain.head())
    # xTest = pd.DataFrame(data=mms.transform(xTest),index=xTest.index,columns=xTest.columns)
    # print(xTest.head())

    space = {'num_leaves': hp.uniform('num_leaves', 2, 2**7),
             'learning_rate': hp.uniform('learning_rate', 0.01, 0.1),
             'n_estimators': hp.randint('n_estimators', 2000),
             'subsample': hp.uniform('subsample', 0.1, 0.98),
             'subsample_freq': hp.uniform('subsample_freq', 1, 5),
             'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 0.98),
             'reg_alpha': hp.uniform('reg_alpha', 0, 1),
             'reg_lambda': hp.uniform('reg_lambda', 0, 1)}

    best_para = fmin(lgbt_optimize, space, algo=tpe.suggest, max_evals=5)

        
    print('')
    print('best_para', best_para)

    print(lgbt_optimize(best_para))

    # 画出优化结果图
    opt_fig()


    # 预测
    df_fu = data_process(file_name, col_name, ds_fu, dt_fu)
    df_X_fu = df_fu.iloc[:, :-1]
    df_y_fu = df_fu.iloc[:, -1]
    fu_pred(best_para, df_X, df_y, df_X_fu, df_y_fu)
























