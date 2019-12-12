# -*- coding: utf-8 -*-

from download_info import download_info
from download_pictures import get_info, get_info_imgs, download

if __name__ == '__main__':
    # 下载图片描述信息
    download_info()
    # 获取图片组信息
    info = get_info()
    # 获取每张图片的url，存储文件夹，本地文件名
    imgs = get_info_imgs(info)
    # 以10个进程并发下载图片
    download(imgs, processes=10)
