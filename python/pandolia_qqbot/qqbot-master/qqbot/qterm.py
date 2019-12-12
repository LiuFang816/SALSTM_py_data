# -*- coding: utf-8 -*-

import sys, socket, time

try:
    import readline
except ImportError:
    pass

from common import CallInNewConsole
from utf8logger import INFO, WARN, RAWINPUT, PRINT
from messagefactory import MessageFactory, Message

HOST, DEFPORT = '127.0.0.1', 8188

class QTermServer:
    def __init__(self, port):
        self.port = port

    def Run(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.bind((HOST, self.port))
            self.sock.listen(5)
        except socket.error as e:
            WARN('无法开启 QQBot term 服务器。%s', e)
        else:
            time.sleep(0.1)
            INFO('已在 %s 端口开启 QQBot-Term 服务器', self.port)
            if CallInNewConsole(['python', __file__, str(self.port)]) != 0:
                WARN('无法自动打开新控制台运行 QTerm 客户端，'
                     '请手动打开新控制台并运行 qterm %s 命令', self.port)

            while True:
                try:
                    sock, addr = self.sock.accept()
                except socket.error:
                    WARN('QQBot-Term 服务器出现 accept 错误')
                else:
                    name = 'QTerm客户端"%s:%s"' % addr
                    sock.settimeout(5.0)
                    try:
                        data = sock.recv(1024)
                    except socket.error:
                        sock.close()
                    else:
                        INFO('QTerm 命令：%s', repr(data))
                        yield TermMessage(name, sock, data)
    
    def processMsg(self, factory, msg):
        if msg.content == 'stop':
            msg.Reply('QQBot已停止')
            factory.Stop()
        else:
            msg.Reply('Hello, ' + msg.content)
    
    def Test(self):
        factory = MessageFactory()
        factory.On('termmessage', self.processMsg) 
        factory.AddGenerator(self.Run)
        factory.Run()

class TermMessage(Message):
    mtype = 'termmessage'

    def __init__(self, name, sock, content):
        self.name = name
        self.sock = sock
        self.content = content

    def Reply(self, rep):
        try:
            self.sock.sendall(rep and str(rep) or '\r\n')
            # INFO('已向 %s 回复消息', self.name)
        except socket.error:
            WARN('回复 %s 失败', self.name)
        finally:
            self.sock.close()

def qterm(port):
    req = 'help'
    while req != 'quit':
        if req:
            resp = query(port, req)    
            if not resp:
                RAWINPUT('与 QQBot term 服务器的连接已断开，按回车键退出')
                break                
            if resp == 'QQBot已停止':
                RAWINPUT('QQBot已停止，按回车键退出')
                break
            resp = resp.strip()
            while True:
                front, resp = partition(resp)
                if resp:
                    RAWINPUT(front+'--More--')
                else:
                    resp = front
                    break
        else:
            resp = ''
        
        if resp:
            req = RAWINPUT(resp+'\nqterm>> ').strip()        
        else:
            req = RAWINPUT('qterm>> ').strip()

def partition(s):
    n = len(s)
    if n <= 800:
        return s, ''
    else:
        for i in range(800, min(n, 900)):
            if s[i] == '\n':
                i += 1
                break
        return s[:i], s[i:]

def query(port, req):
    resp = ''
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:            
        sock.connect((HOST, port))
        sock.sendall(req)
        while True:
            data = sock.recv(8096)
            if not data:
                return resp
            resp += data
    except socket.error:
        return resp
    finally:
        sock.close()

def QTerm():
    # python qterm.py -s
    # python qterm.py [PORT] [COMMAND]
    try:
        if len(sys.argv) == 2 and sys.argv[1] == '-s':
            QTermServer(DEFPORT).Test()
        else:
            if len(sys.argv) >= 2 and sys.argv[1].isdigit():
                port = int(sys.argv[1])
                command = ' '.join(sys.argv[2:])
            else:
                port = DEFPORT
                command = ' '.join(sys.argv[1:])
    
            if not command:
                qterm(port)
            else:
                coding = sys.getfilesystemencoding()
                PRINT(query(port, command.decode(coding).encode('utf8')))
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    QTerm()
