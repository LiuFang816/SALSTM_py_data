import pandas as pd
import static_params
import matplotlib.pyplot as plt
import seaborn as sb

shop_id = '1'

data = pd.read_pickle(static_params.DATA_USER_VIEW_BY_SHOP_PATH + shop_id + '.pkl')

print data
data =  data.groupby(['time']).count()
data.plot()
plt.show()

