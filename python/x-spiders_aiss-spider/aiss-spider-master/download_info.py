# -*- coding: utf-8 -*-
import requests
import json


def download_info():
    """ 下载列表页（包含所有对图片的描述信息），并存储到data/info.txt文件中 """
    page = 1
    while True:
        page_json = download_page(page)
        if not page_json['data']['list']:
            break
        save_page(page_json)
        page += 1


def download_page(page):
    """ 下载某页面的信息 """
    url = 'http://api.pmkoo.cn/aiss/suite/suiteList.do'
    params = {
        'page': page,
        'userId': 153044
    }
    rsp = requests.post(url, data=params)
    return rsp.json()


def save_page(page_json):
    """ 保存某页面的信息 """
    txt = json.dumps(page_json)
    with open('data/info.txt', 'a') as f:
        f.write(txt)
        f.write('\n')


if __name__ == "__main__":
    download_info()
