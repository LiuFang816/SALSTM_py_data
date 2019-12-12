from matplotlib.legend_handler import HandlerLine2D
import numpy as np
import csv
import matplotlib.pyplot as plt
#--------stable-------------------
import os,sys
path = os.getcwd()
parent_path = os.path.dirname(path)
sys.path.append(parent_path)
import static_data as sd
CURRENT_PATH=sd.CURRENT_PATH
ARTIST_FOLDER=sd.ARTIST_FOLDER
ARTIST=sd.ARTIST
SONGS=sd.SONGS
SONG_P_D_C=sd.SONG_P_D_C
ARTIST_P_D_C=sd.ARTIST_P_D_C
SONG_FAN=sd.SONG_FAN
ARTIST_FAN=sd.ARTIST_FAN
DAYS=sd.DAYS
START_UNIX  =sd.START_UNIX
DAY_SECOND  =sd.DAY_SECOND
START_WEEK=sd.START_WEEK
#--------stable-------------------
'''
songs structure:
    {songs1:(mu,sigma),songs2:{mu,sigma},songs3:{mu,sigma}...}
'''
def mean_sigma(songs):
    songs_num=len(songs)
    with open(SONG_P_D_C, "r") as fr:
        songs_id=fr.readline().strip("\n")
        while songs_id and songs_num>0:
            play = list(map(int, fr.readline().strip("\n").split(",")))
            download = list(map(int, fr.readline().strip("\n").split(",")))
            collect = list(map(int, fr.readline().strip("\n").split(",")))
            if songs_id in songs:
                play=np.array(play)
                mu=np.mean(play)
                sigma=np.sqrt((play*play).sum()/DAYS-mu*mu)
                songs[songs_id]=(mu,sigma)
                songs_num-=1
            songs_id=fr.readline().strip("\n")
    return songs

def sum_all():
    return_play=[0 for i in range(DAYS)]
    with open(SONG_P_D_C, "r") as fr:
        songs_id=fr.readline().strip("\n")
        while songs_id:
            play = list(map(int, fr.readline().strip("\n").split(",")))
            download = list(map(int, fr.readline().strip("\n").split(",")))
            collect = list(map(int, fr.readline().strip("\n").split(",")))
            for i in range(DAYS):
                return_play[i]+=play[i]
            songs_id=fr.readline().strip("\n")
    return return_play

'''
songs structure:
    {songs1:True,repeat,repeat,...}
'''
def sum_play(songs):
    songs_num=len(songs)
    return_play=[0 for i in range(DAYS)]
    with open(SONG_P_D_C, "r") as fr:
        songs_id=fr.readline().strip("\n")
        while songs_id and songs_num>0:
            play = list(map(int, fr.readline().strip("\n").split(",")))
            download = list(map(int, fr.readline().strip("\n").split(",")))
            collect = list(map(int, fr.readline().strip("\n").split(",")))
            if songs_id in songs:
                songs_num-=1
                for i in range(DAYS):
                    return_play[i]+=play[i]
            songs_id=fr.readline().strip("\n")
    return return_play


def plot_nor_ms(songs):
    songs_num=len(songs)
    sum_play=np.array([0 for i in range(DAYS)])
    sum_download=np.array([0 for i in range(DAYS)])
    sum_collect=np.array([0 for i in range(DAYS)])
    with open(SONG_P_D_C,'r') as fr:
        songs_id=fr.readline().strip("\n")
        while songs_id and songs_num>0:
            play = list(map(int, fr.readline().strip("\n").split(",")))
            download = list(map(int, fr.readline().strip("\n").split(",")))
            collect = list(map(int, fr.readline().strip("\n").split(",")))
            if songs_id in songs:
                play=np.array(play)
                sum_play+=play
                songs_num-=1
            songs_id=fr.readline().strip("\n")

    p = plt.plot(sum_play, "bo", sum_play, "b-", marker="o")
    #d = plt.plot(download, "ro", download, "r-", marker="o")
    #c = plt.plot(collect, "go", collect, "g-", marker="o")
    #plt.legend([p[1], d[1],c[1]], ["play", "download","collect"])
    plt.lengend(p[1],["play"])
    plt.title('SUM OF THE NORMAL MUSIC')
    plt.xlabel('days')
    plt.ylabel('times')
    #plt.savefig(os.path.join(self.SONG_PLAY_FOLDER, songs_id+".png"))
    plt.show()
