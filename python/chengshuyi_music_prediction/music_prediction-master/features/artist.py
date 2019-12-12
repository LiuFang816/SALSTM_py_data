import song
import numpy as np
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
param(s):
    artists:the artist_id string

return:
    artists:tuple (mu,sigma)
'''
def mean_sigma(artists):
    with open(ARTIST_P_D_C, "r") as fr:
        artist_id = fr.readline().strip("\n")
        while artist_id:
            play = list(map(int, fr.readline().strip("\n").split(",")))
            download = list(map(int, fr.readline().strip("\n").split(",")))
            collect = list(map(int, fr.readline().strip("\n").split(",")))
            if artist_id == artists:
                play=np.array(play)
                mu=np.mean(play)
                sigma=np.sqrt((play*play).sum()/DAYS-mu*mu)
                artists=(mu,sigma)
                break
            artist_id = fr.readline().strip("\n")
    return artists

