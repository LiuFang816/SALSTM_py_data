# -*- coding: utf-8 -*-
import os
import json
import requests
import time


def get_info():
    """ 获取所有图片组的信息 """
    res = []
    with open('data/info.txt', 'r') as f:
        for line in f:
            data = json.loads(line)
            res.extend(data['data']['list'])
    return res


def get_info_imgs(info):
    """ 获取要下载的所有图片url、目录名、要存储的名字 """
    res = []
    for item in info:
        nickname = item["author"]["nickname"]
        catalog = item["source"]["catalog"]
        name = item["source"]["name"]
        issue = item["issue"]
        pictureCount = item["pictureCount"]
        for pic_idx in range(pictureCount):
            url = "http://com-pmkoo-img.oss-cn-beijing.aliyuncs.com/picture/%s/%s/%s.jpg" % (catalog, issue, pic_idx)
            directory = os.path.join("data", name, "%s-%s" % (issue, nickname))
            filepath = os.path.join(directory, "%s.jpg" % pic_idx)
            # 每张图片一组，包含 图片url，所在目录，存储路径
            res.append((
                url, directory, filepath
            ))
    return res


def setup_download_dir(directory):
    """ 设置文件夹，文件夹名为传入的 directory 参数，若不存在会自动创建 """
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except Exception as e:
            pass
    return True

from multiprocessing import Process, Queue, Pool


def download_one(img):
    """ 下载一张图片 """
    url, directory, filepath = img
    # 如果文件已经存在，放弃下载
    if os.path.exists(filepath):
        print('exists:', filepath)
        return

    setup_download_dir(directory)
    rsp = requests.get(url)
    print('start download', url)
    with open(filepath, 'wb') as f:
        f.write(rsp.content)
        print('end download', url)


def download(imgs, processes=10):
    """ 并发下载所有图片 """
    start_time = time.time()
    pool = Pool(processes)
    for img in imgs:
        pool.apply_async(download_one, (img, ))

    pool.close()
    pool.join()
    end_time = time.time()
    print('下载完毕,用时:%s秒' % (end_time - start_time))


if __name__ == "__main__":
    info = get_info()
    imgs = get_info_imgs(info)
    download(imgs, processes=10)
