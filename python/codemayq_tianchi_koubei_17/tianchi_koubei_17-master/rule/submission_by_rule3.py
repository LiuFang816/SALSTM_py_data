# coding=UTF-8
import pandas as pd
import static_params
def get_result_by_last_two_weeks_mean():
    #取前两星期的对应均值，复制两遍
    data = pd.read_pickle(static_params.DATA_PATH + 'user_pay_last_two_weeks.pkl')

    print data

    result = pd.DataFrame(data['iid'])

    date = '2016-11-'
    index = 1
    for index in range(1,8):
        column = date + str(index)
        result[column]  = data.loc[:,['2016-10-' + str(index + 17),'2016-10-' + str(index + 24)]].mean(1)

    data2 = result.copy()
    result = pd.merge(data2,result,on='iid').astype(int)

    result.to_csv(static_params.DATA_PATH + 'submission.csv',header=None,index=None)