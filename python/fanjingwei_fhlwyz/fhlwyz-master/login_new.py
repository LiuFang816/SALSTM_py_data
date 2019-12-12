#!/usr/bin/env python
import httplib, urllib
import socket
import sys, os
from urlparse import urlparse
from getpass import getpass
#httplib.HTTPConnection.debuglevel = 100
marker = "http://www.baidu.com/"

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def lookup_proxy_auth_ip(proxy):
    conn = httplib.HTTPConnection(proxy)
    conn.request("GET", marker)
    response = conn.getresponse()
    conn.close()
    if response.status == httplib.FOUND:
        for item in response.getheaders(): 
          if item[0]=='location':
              return response.status, item[1]
    return response.status, None


proxy = os.getenv("http_proxy") or os.getenv("HTTP_PROXY") or raw_input("Proxy sever: ")
proxy = urlparse(proxy).netloc or proxy

status, auth_url = lookup_proxy_auth_ip(proxy)

if status == httplib.OK:
    print bcolors.OKGREEN + "You already can access %s" % marker+bcolors.ENDC
    sys.exit(0)

account=raw_input("User account: ")
passwd=getpass()

auth_server = urlparse(auth_url).netloc

'''
opr=pwdLogin&userName=10092495&pwd=xxxx&rememberPwd=0&lang=chs
'''

params = urllib.urlencode({
    "lang": "chs",
    "opr" : "pwdLogin",
    "pwd" : passwd,
    "rememberPwd" : "0",
    "userName" : account
})
headers = {"Content-Type": "application/x-www-form-urlencoded"}
conn = httplib.HTTPConnection(auth_server)
conn.request("POST", "/ac_portal/login.php", params, headers)
response = conn.getresponse()
conn.close()
if response.status == httplib.OK:
    print bcolors.OKGREEN+"Login Success. Enjoy your surf! (^.^)"+bcolors.ENDC
