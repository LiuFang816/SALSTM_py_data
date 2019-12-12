import pickle, os
import logging, traceback

import requests

from ..config import VERSION
from ..returnvalues import ReturnValue
from .contact import update_local_chatrooms
from .messages import produce_msg

logger = logging.getLogger('itchat')

def load_hotreload(core):
    core.dump_login_status = dump_login_status
    core.load_login_status = load_login_status

def dump_login_status(self, fileDir=None):
    fileDir = fileDir or self.hotReloadDir
    try:
        with open(fileDir, 'w') as f:
            f.write('itchat - DELETE THIS')
        os.remove(fileDir)
    except:
        raise Exception('Incorrect fileDir')
    status = {
        'version'   : VERSION,
        'loginInfo' : self.loginInfo,
        'cookies'   : self.s.cookies.get_dict(),
        'storage'   : self.storageClass.dumps()}
    with open(fileDir, 'wb') as f:
        pickle.dump(status, f)
    logger.debug('Dump login status for hot reload successfully.')

def load_login_status(self, fileDir,
        loginCallback=None, exitCallback=None):
    try:
        with open(fileDir, 'rb') as f:
            j = pickle.load(f)
    except Exception as e:
        logger.debug('No such file, loading login status failed.')
        return ReturnValue({'BaseResponse': {
            'ErrMsg': 'No such file, loading login status failed.',
            'Ret': -1002, }})

    if j.get('version', '') != VERSION:
        logger.debug(('you have updated itchat from %s to %s, ' + 
            'so cached status is ignored') % (
            j.get('version', 'old version'), VERSION))
        return ReturnValue({'BaseResponse': {
            'ErrMsg': 'cached status ignored because of version',
            'Ret': -1005, }})
    self.loginInfo = j['loginInfo']
    self.s.cookies = requests.utils.cookiejar_from_dict(j['cookies'])
    self.storageClass.loads(j['storage'])
    msgList, contactList = self.get_msg()
    if (msgList or contactList) is None:
        self.logout()
        logger.debug('server refused, loading login status failed.')
        return ReturnValue({'BaseResponse': {
            'ErrMsg': 'server refused, loading login status failed.',
            'Ret': -1003, }})
    else:
        if contactList:
            for contact in contactList:
                if '@@' in contact['UserName']:
                    update_local_chatrooms(self, [contact])
                else:
                    update_local_chatrooms(self, [contact])
        if msgList:
            msgList = produce_msg(self, msgList)
            for msg in msgList: self.msgList.put(msg)
        self.start_receiving(exitCallback)
        logger.debug('loading login status succeeded.')
        if hasattr(loginCallback, '__call__'):
            loginCallback()
        return ReturnValue({'BaseResponse': {
            'ErrMsg': 'loading login status succeeded.',
            'Ret': 0, }})
