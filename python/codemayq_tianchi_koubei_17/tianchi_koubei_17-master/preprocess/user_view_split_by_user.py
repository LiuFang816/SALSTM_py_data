import pandas as pd
import static_params
import cPickle
import os

def get_date(time):
    return time.split(' ')[0]

def user_view_split_by_user():
    if(not os.path.exists(static_params.DATA_USER_VIEW_BY_USER_PATH)):
        os.mkdir(static_params.DATA_USER_VIEW_BY_USER_PATH)

    data = pd.read_pickle(static_params.DATA_PATH + 'user_view.pkl')

    data.columns = ['uid','iid','time']

    data['iid'] = data['iid'].astype(str)
    data['time'] = data['time'].apply(get_date)

    grouped = data.groupby(['uid'],as_index=False)

    for name,group in grouped:
        f = open(static_params.DATA_USER_VIEW_BY_USER_PATH + str(name) + '.pkl', 'wb')
        cPickle.dump(group,f,-1)
        f.close()