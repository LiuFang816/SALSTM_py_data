import pandas as pd
import os
import cPickle
import static_params

def merge_user_view():
    file1 = static_params.DATA_PATH + 'user_view.txt'
    file2 = static_params.DATA_PATH + 'extra_user_view.txt'

    data1 = pd.read_csv(file1,header=None)
    data2 = pd.read_csv(file2,header=None)
    print data1.shape
    print data2.shape

    data = data1.append(data2)

    print data.shape
    f = open(static_params.DATA_PATH + 'user_view.pkl','wb')
    cPickle.dump(data,f,-1)
    f.close()

    os.remove(file1)
    os.remove(file2)