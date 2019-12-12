#coding=utf-8
import time
import itchat
import netease
import threading
import os
import subprocess
import webbrowser
import signal

itchat.auto_login()
will_play_list = []
process = None


def musicbox():
    @itchat.msg_register
    def simple_reply(msg):
        if msg.get('Type', '') == 'Text':
            #return 'I received: %s'%msg.get('Content', '')
            content = msg.get('Content', '')
            content_list = content.split()
            if len(content_list) == 1:
                song_name = content
                musicbox = netease.RasWxMusicbox(song_name)
                music_list = musicbox.gen_music_list()
                return music_list
            if len(content_list) == 2:
                try:
                    song_name = content_list[0]
                    song_index = int(content_list[1])
                    musicbox = netease.RasWxMusicbox(song_name)
                    music_info = musicbox.get_music(song_index)
                    mp3_url = music_info['mp3_url']
                    song_info = u'正在播放:\n ' \
                        + u'专辑： ' + music_info['album_name'] + '\n'\
                        + u'演唱： ' + music_info['artist'] + '\n' \
                        + u'歌曲： ' + music_info['song_name']
                    play(mp3_url)
                    return song_info
                except:
                    return u'输入有误，请重新输入'
            else:
                return u'输入有误，请重新输入'
    itchat.run()


def play(mp3_url):
    try:
        subprocess.Popen(['pkill', 'mpg123'])
        time.sleep(.3)
    except:
        pass
    finally:
        subprocess.Popen(['mpg123', mp3_url])
#        webbrowser.open(mp3_url)

if __name__ == '__main__':
    musicbox()

