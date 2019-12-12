# coding=UTF-8
import pandas as pd
import static_params

#取最后一星期的均值，另外对周五乘以1.1,周六和周日乘以1.2
def get_result_by_last_week_with_weight():
    data = pd.read_pickle(static_params.DATA_USER_PAY_BY_MONTH_PATH + "2016_10.pkl")
    data = data[data['time'] >= '2016-10-25']

    x = data.groupby(['iid'],as_index=False).count()

    x['uid'] = (x['uid']/7).astype(int)

    x['1'] = x['uid']*0.95
    x['2'] = x['uid']*0.95
    x['3'] = x['uid']*0.95
    x['4'] = x['uid']*1.05
    x['5'] = x['uid']*1.12
    x['6'] = x['uid']*1.12
    x['7'] = x['uid']*0.95
    x['8'] = x['uid']*0.95
    x['9'] = x['uid']*0.95
    x['10'] = x['uid']*0.95
    x['11'] = x['uid']*1.05
    x['12'] = x['uid']*1.12
    x['13'] = x['uid']*1.12
    x['14'] = x['uid']*0.95

    x = x.drop(['uid','time'],axis=1).astype(int)

    x.to_csv(static_params.DATA_PATH + 'submission.csv',header=None,index=None)
