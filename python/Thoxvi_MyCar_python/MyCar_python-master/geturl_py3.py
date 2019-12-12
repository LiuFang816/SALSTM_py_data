#!/usr/bin/python3
# coding=utf-8
import sys
import re
import urllib.request
import threading
import time
import sys
import os

newurl = set([])
newusers = set([])



def getpage(url):
	# 下载主页，把主页读取到text
	text = urllib.request.urlopen(url)
	text = text.read().decode('utf-8')
	text = text.split('\n')

    # 寻找框架，并把真正的url放入iframes中
	iframes = []
	for l in text:
		iframe = re.match('.*<iframe src=.(.*). style.*', l)
		if iframe != None:
			iframe = iframe.group(1)
			iframes.append(iframe)

    # 更新用户表
	if len(iframes) != 0:
		newusers.add(re.match('http://(.*)\.tumblr\.com', url).group(1))
	else:
		return None

    # 寻找真正下载地址
	for l in iframes:
		text = urllib.request.urlopen(l)
		text = text.read().decode('utf-8')
		text = text.split('\n')
		for i in text:
			real = re.match('.*<source src=\"(.*)\" type=', i)
			if real != None:
				real = real.group(1)
				oldnum = len(oldurl)  # 去重
				real=re.sub("/480$","",real)  #高清化
				oldurl.add(real)
				if len(oldurl) != oldnum:
					newurl.add(real)
					print(real)
				else:
#					print("已存在")
					pass
				break



# 打开用户名列表
f = open('%s/users' % sys.path[0], 'r')
users = f.read()
f.close()
# 分割和处理并集合化
users = users.split('\n')

#去除 ''
users=list(set(users))
if "" in users:
    users.remove('')

for i in range(0, len(users)):
	users[i] = 'http://%s.tumblr.com' % users[i]
users = set(users)

# myurl.dat
# 导入历来url列表
if os.path.isfile('%s/myurl.dat' % sys.path[0]) == False:
	os.mknod('%s/myurl.dat' % sys.path[0])

f = open('%s/myurl.dat' % sys.path[0], 'r')
oldurl = f.read()
f.close()
# 分割并集合化
oldurl = set(oldurl.split('\n'))

# 多线程并发
ts = []
for i in users:
	th = threading.Thread(target=getpage, args=[i])
	th.start()
	ts.append(th)

for i in ts:
	i.join()

# 向date文件写入新url
f = open('%s/myurl.dat' % sys.path[0], 'a')
for l in newurl:
	f.write("\n%s" % l)
f.close()

# 更新用户表
if len(newusers) != 0:
	f = open('%s/users' % sys.path[0], 'w')
	for l in newusers:
		f.write("\n%s" % l)
	f.close()

# 合并URL
newurl = '\n'.join(newurl)
newurl='Now URL Is Here:\n\n%s\n' % newurl

# 获取时间
nowtime = time.strftime("%m%d%H_%M_%S_%Y", time.localtime())

#写新纪录文件
hispath="%s/HistoryCar"%sys.path[0]
if os.path.isdir(hispath)==False:
	os.mkdir(hispath)
f=open('%s/%s' % (hispath,nowtime), 'w')
f.write(newurl)
f.close()

#write newurl
f=open("%s/newurl"%sys.path[0],"w")
f.write(newurl)
f.close()
