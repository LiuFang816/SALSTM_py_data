# coding=utf-8
import re
import time
import random
import requests
import warnings
from PIL import Image
from StringIO import StringIO

import sys
reload(sys)
CODING = ['utf-8', 'gb2312'][sys.platform in ['win32', 'win64', 'cygwin']]
sys.setdefaultencoding(CODING)


def CheckVersion():
    global AuthorizeList
    warnings.filterwarnings("ignore")
    url = 'https://raw.githubusercontent.com/Lodour/SHU-XK/master/Authorize.txt'
    AuthorizeList = [i for i in re.findall(r'(.*?)\n', requests.get(url, verify=False).content)]

def Login():
    global Username
    Username = raw_input(u'请输入学号: ')
    Password = raw_input(u'请输入密码: ')
    if not Username in AuthorizeList:
        raw_input(u'验证失败...')
        quit()
    print u'正在获取验证码...'
    UrlVerifyPic = UrlIndex + '/Login/GetValidateCode?%20%20+%20GetTimestamp()'
    Image.open(StringIO(Req.get(UrlVerifyPic).content)).show()
    VerifyCode = raw_input(u'请输入验证码: ')
    LoginData = {
        'txtUserName': Username,
        'txtPassword': Password,
        'txtValiCode': VerifyCode
    }
    print u'正在登录...'
    ReqLogin = Req.post(UrlIndex, LoginData)
    Result = re.findall(ur'divLoginAlert">\r\n\s*(.*?)\r\n', ReqLogin.text)
    if len(Result) == 0:
        print u'登录成功!'
        Semester  = re.findall(ur'<font color="red">(.*?)<', ReqLogin.content)[0]
        LoginInfo = re.findall(ur'23px;">\r\n\s*(.*?)：(.*?)\r\n', ReqLogin.text)
        print '%s' % Semester
        for info in LoginInfo:
            print '%s:%s' % info
        return True
    else:
        print u'[登录失败]', Result[0]
        return False

        
def ReadFile():
    File, Line = [], 0
    try:
        F = open('CourseList.txt', 'r')
        for line in F:
            if line[0].isdigit() or line[0].isalpha():
                File.append(line[:13])
        F.close()
        for List in [XKList, TKList]:
            for i in range(int(File[Line])):
                Line += 1
                List.append(File[Line][:13].split(' '))
            print u'\n[%s课列表]' % u'退选'[id(List) == id(XKList)]
            if not List:
                print 'NULL'
            for i in range(len(List)):
                print ['[%d] ','[%d]'][i > 8] % (i + 1),
                print '%s %s' % (List[i][0], List[i][1])
            Line += 1
    except Exception as e:
        print u'[读取选课信息失败] 请检查同目录下CourseList.txt文件'
        print str(e)
        raw_input(u'请按回车后退出...')
        quit()


def Pending():
    UrlXK = UrlIndex + '/CourseSelectionStudent/FastInput'
    ReqXK = Req.get(UrlXK)
    try:
        if u'提示信息' in ReqXK.text:
            Info = re.findall(ur'font-size:24px;">\r\n\s*(.*?)\r\n', ReqXK.text)[0]
            Time = time.strftime('%H:%M:%S', time.localtime(time.time()))
            print '[%s] %s' % (Time, Info)
            return True
    except Exception as e:
        print u'[获取选课系统状态失败] 请手动登录选课系统查询并重新打开本工具'
        print u'[获取选课系统状态失败] 若仍然显示同样信息, 请确认选课系统状态...'
        Check = raw_input(u'选课系统是否已经开启[y/n]:')
        return Check == 'y'
    return False


def XK(Course):
    Data = {
        'stuNo'            : Username,
        'IgnorCredit'      : 'false',
        'IgnorClassMark'   : 'false',
        'IgnorCourseGroup' : 'false',
        'ListCourseStr'    : '%s|%s|0' % (Course[0], Course[1])
    }
    try:
        ReqXK = Req.post(UrlXK, Data, timeout=3)
    except:
        print u'[选课超时]  %s  %s' % (Course[0], Course[1])
        return False
    Success   = u'成功' in ReqXK.text
    Detail    = re.findall(ur'<td>\r\n\s*(.*?)\r\n', ReqXK.text)
    Detail[4] = Detail[4][:4] + u'　' * (4 - len(Detail[4][:4]))
    Output    = [u'[%s]' % Detail[-1], u'[选课成功]'][Success] + '  '
    Output   += u'  '.join([Detail[i][:16] for i in [1, 3, 4, 2]])
    print Output
    if Success or Detail[-1] == u'已选此课程':
        XKList.remove(Course)
    return Success


def TK(Course):
    Data = {
        'StuNo'         : Username,
        'Absolute'      : 'false',
        'ListCourseStr' : '%s|%s' % (Course[0], Course[1])
    }
    try:
        ReqTK = Req.post(UrlTK, Data, timeout=3)
    except:
        print u'[退课超时]  %s  %s' % (Course[0], Course[1])
        return False
    Success   = u'成功' in ReqTK.text
    Detail    = re.findall(ur'<td>\r\n\s*(.*?)\r\n', ReqTK.text)
    Detail[4] = Detail[4][:4] + u'　' * (4 - len(Detail[4][:4]))
    Output    = [u'[%s]' % Detail[-1], u'[退课成功]'][Success] + '  '
    Output   += u'  '.join([Detail[i][:16] for i in [1, 3, 4, 2]])
    print Output
    if Success:
        TKList.remove(Course)
    return Success


if __name__ == '__main__':
    AuthorizeList = []
    CheckVersion()

    Req      = requests.Session()
    UrlIndex = 'http://xk.autoisp.shu.edu.cn:8080'
    UrlXK    = UrlIndex + '/CourseSelectionStudent/CtrlViewOperationResult'
    UrlTK    = UrlIndex + '/CourseReturnStudent/CtrlViewOperationResult'
    Username, Password = '', ''

    while not Login():
        print

    TIMER = int(raw_input(u'请输入选课间隔(s): '))
    XKList, TKList = [], []
    ReadFile()
    raw_input(u'\n按回车键以确认...')

    while Pending():
        RandomSecond = random.uniform(0, 0.5)
        time.sleep(RandomSecond)

    for List in [TKList, XKList]:
        ID = List is XKList
        print u'\n%s课中...' % u'退选'[ID]
        for index in xrange(len(List) - 1, -1, -1):
            [TK, XK][ID](List[index])

    print u'\n刷课中...'
    while XKList:
        for Course in XKList:
            Time = time.strftime('%H:%M:%S', time.localtime(time.time()))
            print '[%s]' % Time,
            XK(Course)
            time.sleep(TIMER)

    raw_input(u'\n已完成, 按回车键后退出...')
