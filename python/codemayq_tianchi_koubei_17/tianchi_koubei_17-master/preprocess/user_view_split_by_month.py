import pandas as pd
import os
import cPickle
import static_params

DATA_DIR = "../data/"
DATA_DIR_SEC = 'user_view_month/'

ISFORMAT = "%Y-%m-%d %H:%M:%S"

def get_hour(time):
    return time.split(' ')[1].split(':')[0]

def user_view_split_by_date():
    if(not os.path.exists(static_params.DATA_USER_VIEW_BY_MONTH_PATH)):
        os.mkdir(static_params.DATA_USER_VIEW_BY_MONTH_PATH)

    data = pd.read_csv(DATA_DIR + 'user_view.txt', header=None)
    data.columns = ['uid', 'iid', 'time']

    data = data[data['time'].astype(str) > '2016-10']

    f = open(static_params.DATA_USER_VIEW_BY_MONTH_PATH + '2016_10' + ".pkl", 'wb')
    cPickle.dump(data, f, -1)
    f.close()
