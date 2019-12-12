# coding=UTF-8
import pandas as pd
import static_params
import matplotlib.pyplot as plt
#要查看的店铺编号
shop_id = '1'

data = pd.read_pickle(static_params.DATA_USER_PAY_BY_SHOP_PATH + shop_id + '.pkl')
#要查看的时期
data = data[data['time'] > '2016-09']

data =  data.groupby(['time']).count()
data.plot()
print data
plt.show()