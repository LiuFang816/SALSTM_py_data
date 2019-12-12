import artist as artist_py
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.pyplot as plt
import song as song_py
import csv

T_T=0.1

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
step:
    read songs of the artist.
    get the mean and sigma of the artist and songs.

param(s):
    artists:the artist_id string.

return:
    return_ms:{ms:(mu,sigma),songs:(mu,sigma),repeat,repeat,...}
'''
def mean_sigma(artists):
    return_ms={}
    return_ms['ms']=artist_py.mean_sigma(artists)

    songs={}
    with open(ARTIST) as csvFile:
        spamreader=csv.reader(csvFile,delimiter=',')
        for row in spamreader:
            if row[1] == artists:
                songs[row[0]]=True
    rtuple=song_py.mean_sigma(songs)
    for i in rtuple:
        return_ms[i]=rtuple[i]
    return return_ms

'''
print the hot music info.
'''
def return_hot_ms(artists):
    hot_ms={}
    return_ms=mean_sigma(artists)
    t=T_T*return_ms['ms'][0]
    for i in return_ms:
        if return_ms[i][0]>t and i!='ms':
            hot_ms[i]=True
            print('song:%-23s mean:%.7f sigma:%.7f'%(i,return_ms[i][0],return_ms[i][1]))
    return hot_ms

'''
print the normal music info.
'''
def return_nor_ms(artists):
    return_ms=mean_sigma(artists)
    t=T_T*return_ms[0][1][0]
    for i in range(1,len(return_ms)):
        if return_ms[i][1][0]<=t:
            print('song:%-23s mean:%.7f sigma:%.7f'%(return_ms[i][0],return_ms[i][1][0],return_ms[i][1][1]))

'''
plot the play times,sum of the hot music.
'''
def plot_for_hot(artists):
    hot_ms=return_hot_ms(artists)
    sum_play=song_py.sum_play(hot_ms)
    p = plt.plot(sum_play, "bo", sum_play, "b-", marker="o")
    #d = plt.plot(download, "ro", download, "r-", marker="o")
    #c = plt.plot(collect, "go", collect, "g-", marker="o")
    #plt.legend([p[1], d[1],c[1]], ["play", "download","collect"])
    plt.title('SUM OF THE HOT MUSIC')
    plt.xlabel('days')
    plt.ylabel('times')
    #plt.savefig(os.path.join(self.SONG_PLAY_FOLDER, songs_id+".png"))
    plt.show()
'''
plot the play times,sum of the normal music.
'''
def plot_for_nor(artists):
    hot_ms=return_hot_ms(artists)
    sum_play=song_py.sum_play(hot_ms)
    all_play=song_py.sum_all()
    for i in range(DAYS):
        sum_play[i]=all_play[i]-sum_play[i]
    p = plt.plot(sum_play, "bo", sum_play, "b-", marker="o")
    #d = plt.plot(download, "ro", download, "r-", marker="o")
    #c = plt.plot(collect, "go", collect, "g-", marker="o")
    #plt.legend([p[1], d[1],c[1]], ["play", "download","collect"])
    plt.title('SUM OF THE NORMAL MUSIC')
    plt.xlabel('days')
    plt.ylabel('times')
    #plt.savefig(os.path.join(self.SONG_PLAY_FOLDER, songs_id+".png"))
    plt.show()

if __name__=='__main__':
    plot_for_hot('0c80008b0a28d356026f4b1097041689')
    plot_for_nor('0c80008b0a28d356026f4b1097041689')
