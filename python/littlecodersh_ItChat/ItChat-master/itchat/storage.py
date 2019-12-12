import os, time, copy
try:
    import Queue
except ImportError:
    import queue as Queue
from threading import Lock

def contact_change(fn):
    def _contact_change(core, *args, **kwargs):
        with core.storageClass.updateLock:
            return fn(core, *args, **kwargs)
    return _contact_change

class Storage(object):
    def __init__(self):
        self.userName          = None
        self.nickName          = None
        self.updateLock        = Lock()
        self.memberList        = []
        self.mpList            = []
        self.chatroomList      = []
        self.msgList           = Queue.Queue(-1)
        self.lastInputUserName = None
    def dumps(self):
        return {
            'userName'          : self.userName,
            'nickName'          : self.nickName,
            'memberList'        : self.memberList,
            'mpList'            : self.mpList,
            'chatroomList'      : self.chatroomList,
            'lastInputUserName' : self.lastInputUserName, }
    def loads(self, j):
        self.userName          = j.get('userName', None)
        self.nickName          = j.get('nickName', None)
        del self.memberList[:]
        for i in j.get('memberList', []): self.memberList.append(i)
        del self.mpList[:]
        for i in j.get('mpList', []): self.mpList.append(i)
        del self.chatroomList[:]
        for i in j.get('chatroomList', []): self.chatroomList.append(i)
        self.lastInputUserName = j.get('lastInputUserName', None)
    def search_friends(self, name=None, userName=None, remarkName=None, nickName=None,
            wechatAccount=None):
        with self.updateLock:
            if (name or userName or remarkName or nickName or wechatAccount) is None:
                return copy.deepcopy(self.memberList[0]) # my own account
            elif userName: # return the only userName match
                for m in self.memberList:
                    if m['UserName'] == userName:
                        return copy.deepcopy(m)
            else:
                matchDict = {
                    'RemarkName' : remarkName,
                    'NickName'   : nickName,
                    'Alias'      : wechatAccount, }
                for k in ('RemarkName', 'NickName', 'Alias'):
                    if matchDict[k] is None:
                        del matchDict[k]
                if name: # select based on name
                    contact = []
                    for m in self.memberList:
                        if any([m.get(k) == name for k in ('RemarkName', 'NickName', 'Alias')]):
                            contact.append(m)
                else:
                    contact = self.memberList[:]
                if matchDict: # select again based on matchDict
                    friendList = []
                    for m in contact:
                        if all([m.get(k) == v for k, v in matchDict.items()]):
                            friendList.append(m)
                    return copy.deepcopy(friendList)
                else:
                    return copy.deepcopy(contact)
    def search_chatrooms(self, name=None, userName=None):
        with self.updateLock:
            if userName is not None:
                for m in self.chatroomList:
                    if m['UserName'] == userName: return copy.deepcopy(m)
            elif name is not None:
                matchList = []
                for m in self.chatroomList:
                    if name in m['NickName']: matchList.append(copy.deepcopy(m))
                return matchList
    def search_mps(self, name=None, userName=None):
        with self.updateLock:
            if userName is not None:
                for m in self.mpList:
                    if m['UserName'] == userName: return copy.deepcopy(m)
            elif name is not None:
                matchList = []
                for m in self.mpList:
                    if name in m['NickName']: matchList.append(copy.deepcopy(m))
                return matchList
