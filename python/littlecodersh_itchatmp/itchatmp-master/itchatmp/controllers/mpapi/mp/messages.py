''' This package is for mass texting in wechat mp
 1. What can we send?
    - IMAGE, VOICE, VIDEO, TEXT, NEWS, CARD

 2. How to send them?
    - we use send_some / send_all method
        `send_some(targetIdList, msgType, mediaId, additionalDict)`
    - for msg like text and card, just pass content as msgId
    - for files like image, voice, video, we need to upload them first
        `upload(fileType, openedFile, additionalDict, permanent)`
    - for news, you need to form them first and upload to get msgId
        `create_news(newsDict, permanent)`
      for images used in news, you need to turn them into url first
        `get_image_url(openedFile)`
    - SPECIAL WARNING: video is a little bit **STRANGE**
      when uploading or sending, you need to pass additionalDict to method
        `{"title" :VIDEO_TITLE, "introduction" :INTRODUCTION}`

 3. I alse listed API list for you:
   - SENDING
     send_some
     send_all
     preview
   - MSG MANAGING
     delete
     get
   - TEMP MATERIAL MANAGING
     upload
     download
   - PERMENENT MATERIAL MANAGING
     get_material
     delete_material
     get_materialcount
     batchget_material
   - FORM NEWS
     create_news
     update_news
     get_image_url
'''
import logging, json, os, mimetypes

from ..requests import requests
from .common import access_token
from itchatmp.utils import retry, encode_send_dict
from itchatmp.config import SERVER_URL
from itchatmp.content import (
    IMAGE, VOICE, VIDEO, THUMB, TEXT, NEWS, CARD)
from itchatmp.returnvalues import ReturnValue

logger = logging.getLogger('itchatmp')

@access_token
def send_some(msgType, mediaId, additionalDict={},
        targetIdList=[], partyIdList=[], tagIdList=[],
        agentId=None, accessToken=None):
    msgDict = __form_send_dict(msgType, mediaId, additionalDict)
    if not msgDict: return msgDict
    if not isinstance(targetIdList, list) or len(targetIdList) < 2:
        return ReturnValue({'errcode': 40130})
    msgDict['touser'] = targetIdList
    r = requests.post('%s/cgi-bin/message/mass/send?access_token=%s' % 
        (SERVER_URL, accessToken), data=encode_send_dict(msgDict))
    def _wrap_result(result):
        return ReturnValue(result.json())
    r._wrap_result = _wrap_result
    return r

@access_token
def send_all(msgType, mediaId, additionalDict={},
        tagId=None, agentId=None, accessToken=None):
    msgDict = __form_send_dict(msgType, mediaId, additionalDict)
    if not msgDict: return msgDict
    if tagId is None: 
        msgDict['filter'] = {'is_to_all': True, 'tag_id': 0}
    else:
        msgDict['filter'] = {'is_to_all': False, 'tag_id': tagId}
    r = requests.post('%s/cgi-bin/message/mass/sendall?access_token=%s' % 
        (SERVER_URL, accessToken), data=encode_send_dict(msgDict))
    def _wrap_result(result):
        result = ReturnValue(result.json())
        if 'media_id' in result: result['errcode'] = 0
        return result
    r._wrap_result = _wrap_result
    return r

@access_token
def preview(msgType, mediaId, additionalDict={},
        toUserId=None, toWxAccount=None, accessToken=None):
    msgDict = __form_send_dict(msgType, mediaId, additionalDict)
    if not msgDict: return msgDict
    if (toUserId or toWxAccount) is None:
        return ReturnValue({'errcode': -10003})
    else:
        if toUserId is not None: msgDict['touser'] = toUserId
        if toWxAccount is not None: msgDict['towxname'] = toWxAccount
    r = requests.post('%s/cgi-bin/message/mass/preview?access_token=%s' % 
        (SERVER_URL, accessToken), data=encode_send_dict(msgDict))
    def _wrap_result(result):
        return ReturnValue(result.json())
    r._wrap_result = _wrap_result
    return r

@access_token
def form_video_id(mediaId, additionalDict, accessToken=None):
    ''' in theory video needs another method to get media_id for sending '''
    additionalDict['media_id'] = mediaId
    additionalDict['description'] = additionalDict['introduction']
    # requests.packages.urllib3.disable_warnings()
    url = 'https://file.api.weixin.qq.com/cgi-bin/media/uploadvideo' \
        '?access_token=%s' % accessToken
    r = requests.post(url, data=encode_send_dict(additionalDict))
        # verify=False).json()
    # I don't know why this is a fake ssl
    def _wrap_result(result):
        result = ReturnValue(result.json())
        if 'media_id' in result: result['errcode'] = 0
        return result
    r._wrap_result = _wrap_result
    return r

def __form_send_dict(msgType, mediaId, additionalDict):
    if not msgType in (IMAGE, VOICE, VIDEO, TEXT, NEWS, CARD):
        return ReturnValue({'errcode': 40004,})
    elif msgType == VIDEO:
        mediaId = form_video_id(mediaId, additionalDict)['media_id']
        if not mediaId: return mediaId
    return {
        NEWS: {'mpnews':{'media_id': mediaId}, 'msgtype': 'mpnews'},
        TEXT: {'text': {'content': mediaId}, 'msgtype': 'text'},
        VOICE: {'voice': {'media_id': mediaId}, 'msgtype': 'voice'},
        IMAGE: {'image': {'media_id': mediaId}, 'msgtype': 'image'},
        VIDEO: {'mpvideo':{'media_id': mediaId,
            'title': additionalDict.get('title', ''),
            'description': additionalDict.get('introduction', '')},
            'msgtype': 'mpvideo'},
        CARD: {'wxcard': {'card_id': mediaId}, 'msgtype': 'wxcard'},
        }[msgType]

@access_token
def delete(msgId, accessToken=None):
    r = requests.post('%s/cgi-bin/message/mass/delete?access_token=%s' % 
        (SERVER_URL, accessToken), data={'msg_id': msgId})
    def _wrap_result(result):
        return ReturnValue(result.json())
    r._wrap_result = _wrap_result
    return r

@access_token
def get(msgId, accessToken=None):
    r = requests.post('%s/cgi-bin/message/mass/get?access_token=%s' % 
        (SERVER_URL, accessToken), data={'msg_id': int(msgId)})
    def _wrap_result(result):
        return ReturnValue(result.json())
    r._wrap_result = _wrap_result
    return r

@access_token
def upload(fileType, fileDir, additionalDict={}, permanent=False, accessToken=None):
    if not fileType in (IMAGE, VOICE, VIDEO, THUMB):
        return ReturnValue({'errcode': 40004,})
    try:
        with open(fileDir, 'rb') as f:
            file_ = f.read()
    except:
        return ReturnValue({'errcode': -10004,})
    fileName = 'file' + os.path.splitext(fileDir)[1].decode('utf8', 'replace')
    fileMime = mimetypes.guess_type(fileName)[0] or 'application/octet-stream'
    if fileType == VIDEO and not ('title' in additionalDict
            and 'introduction' in additionalDict):
        return ReturnValue({'errcode': -10001, 'errmsg': 
            'additionalDict for type VIDEO should be: ' + 
            '{"title" :VIDEO_TITLE, "introduction" :INTRODUCTION}'})
    if permanent:
        url = '%s/cgi-bin/material/add_material?access_token=%s&type=%s'
    else:
        url = '%s/cgi-bin/media/upload?access_token=%s&type=%s' 
    files = {'media': (fileName, file_, fileMime), }
    if fileType == VIDEO:
        files['description'] = (None,
            encode_send_dict(additionalDict), 'application/json'),
    r = requests.post(url % (SERVER_URL, accessToken, fileType),
        files=files)
    def _wrap_result(result):
        result = ReturnValue(result.json())
        if 'media_id' in result: result['errcode'] = 0
        return result
    r._wrap_result = _wrap_result
    return r

@access_token
def download(mediaId, accessToken=None):
    r = requests.get('%s/cgi-bin/media/get?access_token=%s&media_id=%s' % 
        (SERVER_URL, accessToken, mediaId), stream=True)
    def _wrap_result(result):
        if 'application/json' in result.headers['Content-Type']:
            return ReturnValue(result.json())
        else:
            tempStorage = io.BytesIO()
            for block in result.iter_content(1024):
                tempStorage.write(block)
            return ReturnValue({'file': tempStorage, 'errcode': 0})
    r._wrap_result = _wrap_result
    return r

@access_token
def get_material(mediaId, accessToken=None):
    data = {'media_id': mediaId}
    data = encode_send_dict(data)
    if data is None: return ReturnValue({'errcode': -10001})
    r = requests.post('%s/cgi-bin/material/get_material?access_token=%s' % 
        (SERVER_URL, accessToken), data=data, stream=True)
    def _wrap_result(result):
        if 'application/json' in result.headers['Content-Type']:
            j = result.json()
            if 'news_item' in j or 'down_url' in j: j['errcode'] = 0
            return ReturnValue(j)
        else:
            tempStorage = io.BytesIO()
            for block in result.iter_content(1024):
                tempStorage.write(block)
            return ReturnValue({'file': tempStorage, 'errcode': 0})
    r._wrap_result = _wrap_result
    return r

@access_token
def delete_material(mediaId, accessToken=None):
    r = requests.post('%s/cgi-bin/material/del_material?access_token=%s' % 
        (SERVER_URL, accessToken), data={'msg_id': mediaId})
    def _wrap_result(result):
        return ReturnValue(result.json())
    r._wrap_result = _wrap_result
    return r

@access_token
def get_materialcount(accessToken=None):
    r = requests.get('%s/cgi-bin/material/get_materialcount?access_token=%s'
        % (SERVER_URL, accessToken))
    def _wrap_result(result):
        result = ReturnValue(result.json())
        if 'voice_count' in result: result['errcode'] = 0
        return result
    r._wrap_result = _wrap_result
    return r

@access_token
def batchget_material(fileType, offset=0, count=20, accessToken=None):
    if not fileType in (IMAGE, VOICE, VIDEO, THUMB):
        return ReturnValue({'errcode': 40004,})
    if 20 < count: count = 20
    data = {'type': fileType,
        'offset': offset,
        'count': count, }
    data = encode_send_dict(data)
    if data is None: return ReturnValue({'errcode': -10001})
    r = requests.post('%s/cgi-bin/material/batchget_material?access_token=%s'
        % (SERVER_URL, accessToken), data=data)
    def _wrap_result(result):
        result = ReturnValue(result.json())
        if 'total_count' in result: result['errcode'] = 0
        return result
    r._wrap_result = _wrap_result
    return r

@access_token
def create_news(newsDict, permanent=False, accessToken=None):
    if permanent:
        url = '%s/cgi-bin/material/add_news?access_token=%s'
    else:
        url = '%s/cgi-bin/media/uploadnews?access_token=%s'
    r = requests.post(url % (SERVER_URL, accessToken),
        data=encode_send_dict(newsDict))
    def _wrap_result(result):
        result = ReturnValue(result.json())
        if 'media_id' in result: result['errcode'] = 0
        return result
    r._wrap_result = _wrap_result
    return r

@access_token
def update_news(mediaId, newsDict, index=0, accessToken=None):
    data = {
        'media_id': mediaId,
        'index': index,
        'articles': newsDict, }
    data = encode_send_dict(data)
    if data is None: return ReturnValue({'errcode': -10001})
    r = requests.post('%s/cgi-bin/material/update_news?access_token=%s' %
        (SERVER_URL, accessToken), data=data)
    def _wrap_result(result):
        return ReturnValue(result.json())
    r._wrap_result = _wrap_result
    return r

@access_token
def get_image_url(openedFile, accessToken=None):
    r = requests.post('%s/cgi-bin/media/uploadimg?access_token=%s' % 
        (SERVER_URL, accessToken), files={'file': openedFile})
    def _wrap_result(result):
        result = ReturnValue(result.json())
        if 'url' in result: result['errcode'] = 0
        return result
    r._wrap_result = _wrap_result
    return r
