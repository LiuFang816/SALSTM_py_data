#coding=UTF-8
import pandas as pd
import static_params

submission = pd.read_csv(static_params.OUTPUT_PATH + 'submission.csv',header=None)
#观察哪些可能是异常值
print submission[submission.iloc[:,1] < 10]