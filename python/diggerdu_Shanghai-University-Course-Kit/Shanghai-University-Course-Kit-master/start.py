#coding:utf-8

import os
import time

threadNum = 20

for i in range(threadNum):
  os.system("python SHU-XK.py &")
  time.sleep(10.0/threadNum)

