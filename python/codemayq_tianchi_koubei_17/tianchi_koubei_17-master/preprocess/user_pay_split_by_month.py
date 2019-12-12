import pandas as pd
import os
import cPickle
import static_params

DATA_DIR = "../data/"
DATA_DIR_SEC = 'user_pay_month/'

ISFORMAT = "%Y-%m-%d %H:%M:%S"


def user_pay_split_by_month():
    if(not os.path.exists(static_params.DATA_USER_PAY_BY_MONTH_PATH)):
        os.mkdir(static_params.DATA_USER_PAY_BY_MONTH_PATH)

    data = pd.read_csv(static_params.DATA_PATH + 'user_pay.txt', header=None)
    data.columns = ['uid', 'iid', 'time']

    data = data[data['time'].astype(str) > '2016-10']

    f = open(static_params.DATA_USER_PAY_BY_MONTH_PATH + '2016_10' + ".pkl", 'wb')
    cPickle.dump(data, f, -1)
    f.close()
