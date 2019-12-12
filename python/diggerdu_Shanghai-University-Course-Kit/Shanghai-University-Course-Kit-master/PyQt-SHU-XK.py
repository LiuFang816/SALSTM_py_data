# coding=utf-8
import os, sys
import re
import time
import requests
from PyQt4.QtGui import *
from PyQt4.QtCore import *
import warnings

def FormatLogging(str):
    Time = time.strftime('<b>%H:%M:%S</b>', time.localtime(time.time()))
    return u'%s  %s' % (Time, str)


class ThreadOfGetPicture(QThread):
    """后台获取并更新验证码"""
    trigger = pyqtSignal(requests.models.Response)

    def __init__(self, parent=None):
        super(ThreadOfGetPicture, self).__init__(parent)
   
    def run(self):
        UrlValidPic = INDEX_URL + '/Login/GetValidateCode?%20%20+%20GetTimestamp()'
        self.trigger.emit(REQ.get(UrlValidPic))


class ThreadOfXK(QThread):
    """后台选课子进程"""
    trigger = pyqtSignal(str)

    def __init__(self, parent, req):
        QThread.__init__(self, parent)
        self.req = req

    def run(self):
        global COURSE_LIST
        while COURSE_LIST.count([None, None]) != len(COURSE_LIST):
            for i in COURSE_LIST:
                if not i[0]:
                    continue
                Result = self.XK(i)
                if Result[0] == True:
                    COURSE_LIST[COURSE_LIST.index(i)] = [None, None]
                echo = u'%s:%s  %s' % (i[0], i[1], Result[1])
                self.trigger.emit(FormatLogging(echo))
                time.sleep(TIMER * 0.001)

    def XK(self, Course):
        Data = {
            'stuNo'            : USERNAME,
            'IgnorCredit'      : 'false',
            'IgnorClassMark'   : 'false',
            'IgnorCourseGroup' : 'false',
            'ListCourseStr'    : '%s|%s|0' % (Course[0], Course[1])
        }
        UrlXK = INDEX_URL + '/CourseSelectionStudent/CtrlViewOperationResult'
        try:
            ReqXK = self.req.post(UrlXK, Data, timeout=10)
        except Exception as e:
            # QMessageBox.critical(self, u'遇到错误', str(e))
            return [False, u'选课超时']
        try:
            Success   = u'成功' in ReqXK.text
            Detail    = re.findall(ur'<td>\r\n\s*(.*?)\r\n', ReqXK.text)
            Detail[4] = Detail[4][:4] + u'　' * (4 - len(Detail[4][:4]))
            Output    = [u'[%s]' % Detail[-1], u'[选课成功]'][Success] + '  '
            Output   += u'  '.join([Detail[i][:16] for i in [1, 3, 4, 2]])
            if Success:
                return [True, u'<font color="red">选课成功</font>']
            if Detail[-1] == u'已选此课程':
                return [True, u'<font color="orange">已选此课程</font>']
            return [False, u'<font color="green">%s</font>' % Detail[-1]]
        except Exception as e:
            return [False, u'<b>未知错误:</b>%s' % e]


class ThreadOfStart(QThread):
    """后台选课主线程"""
    trigger = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, parent=None):
        super(ThreadOfStart, self).__init__(parent)

    def run(self):
        self.threads = []
        for req in REQS:
            if COURSE_LIST.count([None, None]) == len(COURSE_LIST):
                self.finished.emit()
            if req[1] == False:
                continue
            thread = ThreadOfXK(None, req[0])
            thread.trigger.connect(self.UpdateLogging)
            thread.start()
            self.threads.append(thread)
            time.sleep(TIMER * 0.001 / len(REQS))

    def UpdateLogging(self, log):
        self.trigger.emit(log)


class XK_LoginWindow(QDialog):
    """确认选课学期的窗口"""

    # 初始化
    def __init__(self, parent=None):
        # 调用父类初始化
        QDialog.__init__(self, parent)

        # 固定窗口大小
        self.setFixedSize(425, 200)

        # 设置标题
        self.setWindowTitle(u'SHU选课助手 - By Mango Gao')

        # 设置小图标
        icon = QIcon()
        icon.addPixmap(QPixmap("QQHead.ico"),QIcon.Normal, QIcon.Off)
        self.setWindowIcon(icon)

        # 创建布局组件
        Grid = QGridLayout()

        # 创建并添加提示标签，占1行2列
        self.lblTip = QLabel(u'学　期')
        Grid.addWidget(self.lblTip, 0, 0)

        # 创建并添加两个学期的单选框，分别占1行2列
        self.Semester = self.GetSemester()
        self.rdoSemester1 = QRadioButton(self.Semester[0][0])
        self.rdoSemester2 = QRadioButton(self.Semester[0][1])
        Grid.addWidget(self.rdoSemester1, 0, 1)
        Grid.addWidget(self.rdoSemester2, 0, 2)

        # 创建并添加3个提示标签 居中
        self.lblUsername = QLabel(u'学　号')
        self.lblPassword = QLabel(u'密　码')
        self.lblValidNum = QLabel(u'验证码')
        self.lblUsername.setAlignment(Qt.AlignCenter)
        self.lblPassword.setAlignment(Qt.AlignCenter)
        self.lblValidNum.setAlignment(Qt.AlignCenter)
        Grid.addWidget(self.lblUsername, 1, 0)
        Grid.addWidget(self.lblPassword, 2, 0)
        Grid.addWidget(self.lblValidNum, 3, 0)

        # 创建并添加3个输入框 设置默认文字 设置密码不显示明文
        self.leUsername = QLineEdit()
        self.lePassword = QLineEdit()
        self.leValidNum = QLineEdit()
        self.leUsername.setPlaceholderText(u'请输入学号...')
        self.lePassword.setPlaceholderText(u'请输入密码...')
        self.leValidNum.setPlaceholderText(u'请输入验证码...')
        self.lePassword.setEchoMode(QLineEdit.Password)
        Grid.addWidget(self.leUsername, 1, 1, 1, 2)
        Grid.addWidget(self.lePassword, 2, 1, 1, 2)
        Grid.addWidget(self.leValidNum, 3, 1)

        # 创建并添加验证码图片
        self.lblValidPic = QLabel()
        Grid.addWidget(self.lblValidPic, 3, 2)

        # 设置更新学期的函数
        self.rdoSemester1.clicked.connect(lambda: self.UpdateSemester(1))
        self.rdoSemester2.clicked.connect(lambda: self.UpdateSemester(2))

        # 默认选中第一个学期
        self.rdoSemester1.setChecked(True)
        self.UpdateSemester(1)

        # 创建并添加登录按钮
        self.pbLogin = QPushButton(u'登　录')
        Grid.addWidget(self.pbLogin, 4, 0, 1, 3)

        # 设置pbLogin的函数
        self.pbLogin.clicked.connect(self.Login)

        # 设置布局组件
        self.setLayout(Grid)

        # 授权名单
        warnings.filterwarnings("ignore")
        url = 'http://raw.githubusercontent.com/Lodour/SHU-XK/master/Authorize.txt'
        self.UserList = [i for i in re.findall(r'(.*?)\n', requests.get(url, verify=False).content)]

    # 获取学期信息
    def GetSemester(self):
        Url = ['http://xk.autoisp.shu.edu.cn',
               'http://xk.autoisp.shu.edu.cn:8080']
        Sem = map(lambda i: re.findall(ur'center;">(.*?)<', REQ.get(Url[i]).text)[0], range(len(Url)))
        return [Sem, Url]

    # 选择学期后的处理
    def UpdateSemester(self, op=None):
        global INDEX_URL

        # op传值时，更新学期
        if not op == None:
            INDEX_URL = self.Semester[1][op - 1]

        # 建立子进程获取验证码
        self.threads = []
        thread = ThreadOfGetPicture(self)
        thread.trigger.connect(self.ReflashPic)
        thread.start()
        self.threads.append(thread)

    # 刷新验证码
    def ReflashPic(self, req):
        PixValidPic = QPixmap()
        PixValidPic.loadFromData(req.content)
        self.lblValidPic.setPixmap(PixValidPic)
        self.leValidNum.clear()

    # 登录
    def Login(self):

        # 配置登录信息，QStirng -> string
        LoginData = {
            'txtUserName': str(self.leUsername.text()),
            'txtPassword': str(self.lePassword.text()),
            'txtValiCode': str(self.leValidNum.text())
        }

        # 授权
        if not LoginData['txtUserName'] in self.UserList:
            QMessageBox.critical(self, u'验证失败', u'你不在使用名单中哦~')
            return

        # 登录并获取返回错误结果
        ReqLogin = REQ.post(INDEX_URL, LoginData)
        Result = re.findall(ur'divLoginAlert">\r\n\s*(.*?)\r\n', ReqLogin.text)

        # 处理结果
        if not Result:

            # 动态获取学期、个人信息
            Semester = re.findall(ur'<font color="red">(.*?)<', ReqLogin.text)[0]
            LoginInfo = re.findall(ur'23px;">\r\n\s*(.*?)：(.*?)\r\n', ReqLogin.text)

            # 加载并显示输出信息
            EchoInfo = u'<p>登陆成功</p>'
            EchoInfo += u'<p>当前学期: %s</p>' % Semester
            for info in LoginInfo:
                EchoInfo += u'<p>%s: %s</p>' % info
            QMessageBox.information(self, u'登录成功', EchoInfo)

            # 设置全局变量：学号和密码
            global USERNAME, PASSWORD
            USERNAME, PASSWORD = LoginData['txtUserName'], LoginData['txtPassword']
            REQ.get(INDEX_URL + '/CourseSelectionStudent/FastInput')
            REQS.append([REQ, True])

            # 关闭登录窗口
            self.accept()

        else:

            # 加载并显示输出信息
            EchoInfo = u'<p>登陆失败</p>'
            EchoInfo += Result[0]
            QMessageBox.critical(self, u'登录失败', EchoInfo)
            self.UpdateSemester()
        

class XK_Setting(QDialog):
    """设置窗口"""

    # 按下[添加任务]的信号
    pbConfirmClicked = pyqtSignal(str)

    # 更新Timer时的信号
    TimerChanged = pyqtSignal(str)

    # 初始化
    def __init__(self, parent=None):
        # 调用父类初始化
        QDialog.__init__(self, parent)

        # 固定窗口大小
        self.setFixedSize(225, 325)

        # 设置标题
        self.setWindowTitle(u'SHU选课助手')

        # 设置小图标
        icon = QIcon()
        icon.addPixmap(QPixmap("QQHead.ico"),QIcon.Normal, QIcon.Off)
        self.setWindowIcon(icon)

        # 创建布局组件
        Grid = QGridLayout()

        # 设置延时
        self.lblTimer = QLabel(u'延时(ms):')
        self.sbTimer = QSpinBox()
        self.sbTimer.setRange(1000, 10000)
        self.sbTimer.setSingleStep(100)
        self.sbTimer.setValue(TIMER)
        Grid.addWidget(self.lblTimer, 0, 0)
        Grid.addWidget(self.sbTimer, 0, 1)
        self.sbTimer.valueChanged.connect(self.ChangeTimer)

        # 添加任务
        self.leCourseID = QLineEdit()
        self.leTeacheID = QLineEdit()
        self.leCourseID.setPlaceholderText(u'课程号...')
        self.leTeacheID.setPlaceholderText(u'教师号...')
        self.pbConfirm = QPushButton(u'添加任务')
        self.pbConfirm.setDefault(False)
        Grid.addWidget(self.leCourseID, 1, 0)
        Grid.addWidget(self.leTeacheID, 1, 1)
        Grid.addWidget(self.pbConfirm, 2, 0, 1, 2)

        # cookies
        self.lblCookies = QLabel()
        self.lwCookies = QListWidget()
        Grid.addWidget(self.lblCookies, 3, 0, 1, 2)
        Grid.addWidget(self.lwCookies, 4, 0, 3, 2)
        self.lwCookies.setSelectionMode(QAbstractItemView.SingleSelection)
        # 添加REQS中的cookie
        for req in REQS:
            if req[1]:
                SessionID = requests.utils.dict_from_cookiejar(req[0].cookies)['ASP.NET_SessionId']
                self.lwCookies.addItem(SessionID)
        self.lblCookies.setText(u'Cookie列表(%d)' % self.lwCookies.count())

        # 验证码
        self.leValidCode = QLineEdit()
        self.lblValidPic = QLabel()
        Grid.addWidget(self.leValidCode, 7, 0)
        Grid.addWidget(self.lblValidPic, 7, 1)
        self.connect(self.leValidCode, SIGNAL("returnPressed()"), self.Login)

        # 如果最后一个会话已经登录，则添加新的会话
        if REQS[-1][1]:
            REQS.append([requests.Session(), False])
        # 用最后一个会话更新验证码
        self.ReflashPic()


        # 登录按钮
        self.pbLogin = QPushButton(u'登录以添加Cookie')
        Grid.addWidget(self.pbLogin, 8, 0, 1, 2)

        # 添加任务的函数
        self.pbConfirm.clicked.connect(self.Confirm)

        # 登录账号的函数
        self.pbLogin.clicked.connect(self.Login)

        # 设置布局组件
        self.setLayout(Grid)

    def ChangeTimer(self, num):
        global TIMER
        TIMER = num
        self.TimerChanged.emit(u'>> 更新延时: %d(ms)' % TIMER)

    def Confirm(self):
        cID = str(self.leCourseID.text())
        tID = str(self.leTeacheID.text())
        if len(cID) == 8 and len(tID) == 4:
            echo = u'>> 添加任务: %s %s' % (cID, tID)
            self.pbConfirmClicked.emit(echo)
            COURSE_LIST.append([cID, tID])
            self.leCourseID.clear()
            self.leTeacheID.clear()

    # 用REQS[-1]刷新验证码
    def ReflashPic(self):
        UrlValidPic = INDEX_URL + '/Login/GetValidateCode?%20%20+%20GetTimestamp()'
        ReqValidPic = REQS[-1][0].get(UrlValidPic)
        PixValidPic = QPixmap()
        PixValidPic.loadFromData(ReqValidPic.content)
        self.lblValidPic.setPixmap(PixValidPic)
        self.leValidCode.clear()

    def Login(self):
        ValidNum = str(self.leValidCode.text())
        if not ValidNum:
            QMessageBox.critical(self, u'验证码出错', u'请输入验证码！')
            return False
        LoginData = {
            'txtUserName': USERNAME,
            'txtPassword': PASSWORD,
            'txtValiCode': ValidNum
        }
        req = REQS[-1][0]
        req.get(INDEX_URL)
        ReqLogin = req.post(INDEX_URL, LoginData)
        Result = re.findall(ur'divLoginAlert">\r\n\s*(.*?)\r\n', ReqLogin.text)
        if not Result:
            SessionID = requests.utils.dict_from_cookiejar(req.cookies)['ASP.NET_SessionId']
            self.lwCookies.addItem(SessionID)
            self.lblCookies.setText(u'Cookie列表(%d)' % self.lwCookies.count())
            req.get(INDEX_URL + '/CourseSelectionStudent/FastInput')
            REQS[-1][1] = True
            REQS.append([requests.Session(), False])
            REQS[-1][0].get(INDEX_URL)
            self.ReflashPic()
        else:
            EchoInfo = u'<p>登陆失败</p>' + Result[0]
            QMessageBox.critical(self, u'登录失败', EchoInfo)
            self.ReflashPic()


class XK_MainWindow(QDialog):
    """选课的主窗口"""

    # 初始化
    def __init__(self, parent=None):
        # 调用父类初始化
        QDialog.__init__(self, parent)

        # 固定窗口大小
        self.setFixedSize(350, 300)

        # 设置标题
        self.setWindowTitle(u'SHU选课助手 - %s' % USERNAME)

        # 设置小图标
        icon = QIcon()
        icon.addPixmap(QPixmap("QQHead.ico"),QIcon.Normal, QIcon.Off)
        self.setWindowIcon(icon)

        # 创建布局组件
        Grid = QGridLayout()

        # 信息窗口
        self.tbLogging = QTextBrowser()
        self.tbLogging.append(u'<b>SHU选课助手 Version 2.0</b>')
        self.tbLogging.append(u'<b>Author: </b>Mango Gao')
        self.tbLogging.setReadOnly(1)
        Grid.addWidget(self.tbLogging, 0, 0, 4, 2)

        # 设置按钮
        self.pbSetting = QPushButton(u'设置')
        Grid.addWidget(self.pbSetting, 4, 0)

        # 开始按钮
        self.pbStart = QPushButton(u'开始')
        Grid.addWidget(self.pbStart, 4, 1)

        # 设置布局组件
        self.setLayout(Grid)

        # pbSetting的函数
        self.pbSetting.clicked.connect(self.Setting)

        # pbStart的函数
        self.pbStart.clicked.connect(self.Start)

    def Setting(self):
        self.SetWindow = XK_Setting()
        self.SetWindow.pbConfirmClicked.connect(self.AddMission)
        self.SetWindow.TimerChanged.connect(self.AddMission)
        self.SetWindow.show()

    def AddMission(self, echo):
        self.tbLogging.append(u'<font color="blue">%s</font>' % echo)

    def Start(self):
        global WORKING
        self.pbSetting.setEnabled(WORKING)
        WORKING = not WORKING
        self.pbStart.setText([u'开始', u'停止'][WORKING])
        if WORKING:
            self.tbLogging.append(u'<font color="blue">>> 开始刷课</font>')
            self.threads = []
            thread = ThreadOfStart(self)
            thread.trigger.connect(self.UpdateLogging)
            thread.finished.connect(self.Finished)
            thread.start()
            self.threads.append(thread)
        else:
            for i in self.threads:
                for j in i.threads:
                    j.terminate()
                i.terminate()
            self.tbLogging.append(u'<font color="blue">>> 停止刷课</font>')


    def UpdateLogging(self, log):
        self.tbLogging.append(log)

    def Finished(self):
        for i in self.threads:
            i.terminate()
        self.tbLogging.append(u'<font color="blue">>> 已完成</font>')
        self.pbStart.setText(u'开始')
        self.pbSetting.setEnabled(True)
        global WORKING
        WORKING = False


if __name__ == '__main__':
    INDEX_URL = ''
    REQ, REQS = requests.Session(), []
    USERNAME, PASSWORD = '', ''
    COURSE_LIST = []
    TIMER = 6300
    WORKING = False

    # 主应用
    SHU_XK = QApplication(sys.argv)

    # 选择学期并登录
    LoginWindow = XK_LoginWindow()
    if LoginWindow.exec_():
        # 进入选课窗口
        MainWindow = XK_MainWindow()
        MainWindow.show()
    else:
        sys.exit()

    # 执行应用
    SHU_XK.exec_()
