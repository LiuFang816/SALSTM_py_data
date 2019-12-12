# coding=UTF-8
import pandas as pd
import static_params

def get_result_by_last_three_weeks_mean():
    data = pd.read_pickle(static_params.DATA_PATH + 'user_pay_last_three_weeks.pkl')

    result = pd.DataFrame(data['iid'])

    date = '2016-11-'
    index = 1
    for index in range(1,8):
        column = date + str(index)
        result[column]  = data.loc[:,['2016-10-' + str(index + 10),'2016-10-' + str(index + 17),'2016-10-' + str(index + 24)]].mean(1)

    data2 = result.copy()
    result = pd.merge(data2,result,on='iid')

    result.iloc[:,-4] = result.iloc[:,-4]*1.2
    result = result.astype(int)

    result.to_csv(static_params.DATA_PATH + 'submission.csv',header=None,index=None)