import pandas as pd
import numpy as np
import cPickle
import static_params
def get_date(time):
    return time.split(' ')[0]

def user_pay_get_last_two_weeks():
    data = pd.read_pickle(static_params.DATA_USER_PAY_BY_MONTH_PATH + "2016_10.pkl")

    data = data[data['time'] >= '2016-10-18']
    data['time'] = data['time'].apply(get_date)

    data = data.groupby(['iid','time'],as_index=False).count()

    result = pd.DataFrame(np.arange(1,2001),columns=['iid'])

    for index in range(18,32):
        date = '2016-10-' + str(index)
        result[date] = np.zeros((result.shape[0],1))

    for row in data.values:
        result.loc[row[0] - 1,row[1]] = row[2]

    shape = result.shape
    for x in range(shape[0]):
        for index in range(1, shape[1]):
            median = result.iloc[x, 1:].median()
            if (result.iloc[x, index] <= 1):
                result.iloc[x, index] = median
            if (result.iloc[x, index] <= median/5):
                result.iloc[x, index] = median

    f = open(static_params.DATA_PATH + "user_pay_last_two_weeks.pkl", 'wb')
    cPickle.dump(result.astype(int), f, -1)
    f.close()
