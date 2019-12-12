import shlex,json,os,zipfile,glob,logging
import subprocess as sp
import indexer
import numpy as np
import pyscenecustom

class WQuery(object):

    def __init__(self,dquery,media_dir):
        self.media_dir = media_dir
        self.dquery = dquery
        self.primary_key = self.dquery.pk
        self.local_path = "{}/queries/{}.png".format(self.media_dir,self.primary_key)

    def find(self,n=10):
        results = {}
        for index_name,index in indexer.INDEXERS.iteritems():
            results[index_name] = []
            index.load_index(path=self.media_dir)
            results[index_name] = index.nearest(image_path=self.local_path,n=n)
        return results

class WVideo(object):

    def __init__(self,dvideo,media_dir,rescaled_width=600):
        self.dvideo = dvideo
        self.primary_key = self.dvideo.pk
        self.media_dir = media_dir
        self.local_path = "{}/{}/video/{}.mp4".format(self.media_dir,self.primary_key,self.primary_key)
        self.duration = None
        self.width = None
        self.height = None
        self.rescaled_width = rescaled_width
        self.metadata = {}

    def get_metadata(self):
        if self.dvideo.youtube_video:
            output_dir = "{}/{}/{}/".format(self.media_dir,self.primary_key,'video')
            command = "youtube-dl -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4'  \"{}\" -o {}.mp4".format(self.dvideo.url,self.primary_key)
            logging.info(command)
            download = sp.Popen(shlex.split(command),cwd=output_dir)
            download.wait()
            if download.returncode != 0:
                logging.error("Could not download the video")
                raise ValueError
        command = ['ffprobe','-i',self.local_path,'-print_format','json','-show_format','-show_streams','-v','quiet']
        p = sp.Popen(command,stdout=sp.PIPE,stderr=sp.STDOUT,stdin=sp.PIPE)
        p.wait()
        output, _ = p.communicate()
        self.metadata = json.loads(output)
        try:
            self.duration = float(self.metadata['format']['duration'])
            self.width = float(self.metadata['streams'][0]['width'])
            self.height = float(self.metadata['streams'][0]['height'])
        except:
            raise ValueError,str(self.metadata)

    def extract_frames(self):
        frames = []
        if not self.dvideo.dataset:
            output_dir = "{}/{}/{}/".format(self.media_dir,self.primary_key,'frames')
            command = 'ffmpeg -i {} -vf "select=not(mod(n\,100)),scale={}:-1" -vsync vfr  {}/%d_b.jpg'.format(self.local_path,self.rescaled_width,output_dir)
            extract = sp.Popen(shlex.split(command))
            extract.wait()
            if extract.returncode != 0:
                raise ValueError
            for fname in glob.glob(output_dir+'*_b.jpg'):
                ind = int(fname.split('/')[-1].replace('_b.jpg', ''))
                os.rename(fname,fname.replace('{}_b.jpg'.format(ind),'{}.jpg'.format(ind*100)))
                f = WFrame(frame_index=int(100*ind),video=self)
                if extract.returncode != 0:
                    raise ValueError
                frames.append(f)
            self.scene_detection(frames)
        else:
            zipf = zipfile.ZipFile("{}/{}/video/{}.zip".format(self.media_dir, self.primary_key, self.primary_key), 'r')
            zipf.extractall("{}/{}/frames/".format(self.media_dir, self.primary_key))
            zipf.close()
            i = 0
            for subdir, dirs, files in os.walk("{}/{}/frames/".format(self.media_dir, self.primary_key)):
                if '__MACOSX' not in subdir:
                    for fname in files:
                        fname = os.path.join(subdir,fname)
                        if fname.endswith('jpg') or fname.endswith('jpeg'):
                            i += 1
                            dst = "{}/{}/frames/{}.jpg".format(self.media_dir, self.primary_key, i)
                            os.rename(fname, dst)
                            f = WFrame(frame_index=i, video=self,name=fname.split('/')[-1],
                                       subdir=subdir.replace("{}/{}/frames/".format(self.media_dir, self.primary_key),'')
                                       )
                            frames.append(f)
                        else:
                            logging.warning("skipping {} not a jpeg file".format(fname))
                else:
                    logging.warning("skipping {} ".format(subdir))
        return frames

    def index_frames(self,frames):
        results = []
        wframes = [WFrame(video=self, frame_index=df.frame_index,primary_key=df.pk) for df in frames]
        for index_name,index in indexer.INDEXERS.iteritems():
            index.load()
            results.append(index.index_frames(wframes,self))
        return results

    def scene_detection(self,frames):
        manager = pyscenecustom.manager.SceneManager(save_image_prefix="{}/{}/frames/".format(self.media_dir,self.primary_key),rescaled_width=self.rescaled_width)
        path = self.local_path
        framelist = pyscenecustom.detect_scenes_file(path, manager)
        for s in framelist:
            f = WFrame(frame_index=s, video=self)
            frames.append(f)
        return frames


class WFrame(object):

    def __init__(self,frame_index=None,video=None,primary_key=None,name=None,subdir=None):
        if video:
            self.subdir = subdir
            self.frame_index = frame_index
            self.video = video
            self.primary_key = primary_key
            self.name = name
        else:
            self.subdir = None
            self.frame_index = None
            self.video = None
            self.primary_key = None
            self.name = None

    def local_path(self):
        return "{}/{}/{}/{}.jpg".format(self.video.media_dir,self.video.primary_key,'frames',self.frame_index)

