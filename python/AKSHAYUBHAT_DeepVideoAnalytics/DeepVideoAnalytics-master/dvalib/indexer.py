import numpy as np
import os,glob,logging
import torch
import PIL
from torch.autograd import Variable
from torchvision import transforms
from torchvision.models import alexnet
# import alexnet
from scipy import spatial
import tensorflow as tf
from scipy import spatial
from tensorflow.python.platform import gfile


class BaseIndexer(object):

    def __init__(self):
        self.name = "base"
        self.net = None
        self.indexed_dirs = set()
        self.index, self.files, self.findex = None, {}, 0


    def load_index(self,path):
        temp_index = []
        for dirname in os.listdir(path +"/"):
            fname = "{}/{}/indexes/{}.npy".format(path,dirname,self.name)
            if dirname not in self.indexed_dirs and dirname != 'queries' and os.path.isfile(fname):
                logging.info("Starting {}".format(fname))
                self.indexed_dirs.add(dirname)
                try:
                    t = np.load(fname)
                    if max(t.shape) > 0:
                        temp_index.append(t)
                    else:
                        raise ValueError
                except:
                    logging.error("Could not load {}".format(fname))
                    pass
                else:
                    for i, f in enumerate(file(fname.replace(".npy", ".framelist")).readlines()):
                        frame_index,frame_pk = f.strip().split('_')
                        self.files[self.findex] = {
                            'frame_index':frame_index,
                            'video_primary_key':dirname,
                            'frame_primary_key':frame_pk
                        }
                        # ENGINE.store_vector(index[-1][i, :], "{}".format(findex))
                        self.findex += 1
                    logging.info("Loaded {}".format(fname))
        if self.index is None:
            self.index = np.concatenate(temp_index)
            self.index = self.index.squeeze()
            logging.info(self.index.shape)
        elif temp_index:
            self.index = np.concatenate([self.index, np.concatenate(temp_index).squeeze()])
            logging.info(self.index.shape)

    def nearest(self,image_path,n=12):
        query_vector = self.apply(image_path)
        temp = []
        dist = []
        logging.info("started query")
        for k in xrange(self.index.shape[0]):
            temp.append(self.index[k])
            if (k+1) % 50000 == 0:
                temp = np.transpose(np.dstack(temp)[0])
                dist.append(spatial.distance.cdist(query_vector,temp))
                temp = []
        if temp:
            temp = np.transpose(np.dstack(temp)[0])
            dist.append(spatial.distance.cdist(query_vector,temp))
        dist = np.hstack(dist)
        ranked = np.squeeze(dist.argsort())
        logging.info("query finished")
        results = []
        for i, k in enumerate(ranked[:n]):
            temp = {'rank':i,'algo':self.name,'dist':dist[0,k]}
            temp.update(self.files[k])
            results.append(temp)
        return results

    def index_frames(self,frames,video):
        files = []
        features = []
        media_dir = video.media_dir
        for f in frames:
            files.append("{}_{}".format(f.frame_index,f.primary_key))
            features.append(self.apply(f.local_path()))
        feat_fname = "{}/{}/indexes/{}.npy".format(media_dir,video.primary_key,self.name)
        files_fname = "{}/{}/indexes/{}.framelist".format(media_dir, video.primary_key,self.name)
        with open(feat_fname, 'w') as feats:
            np.save(feats, np.array(features))
        with open(files_fname, 'w') as filelist:
            filelist.write("\n".join(files))
        return {'index_name':self.name,'count':len(features)}


class AlexnetIndexer(BaseIndexer):

    def __init__(self):
        self.name = "alexnet"
        self.net = None
        self.transform = None
        self.indexed_dirs = set()
        self.index, self.files, self.findex = None, {}, 0

    def apply(self,path):
        self.load()
        tensor = self.transform(PIL.Image.open(path).convert('RGB')).unsqueeze_(0)
        if torch.cuda.is_available():
            tensor = torch.FloatTensor(tensor).cuda()
        result = self.net(Variable(tensor))
        if torch.cuda.is_available():
            return result.data.cpu().numpy()
        return result.data.numpy()


    def load(self):
        if self.net is None:
            logging.warning("Loading the network {}".format(self.name))
            self.net = alexnet(pretrained=True)
            if torch.cuda.is_available():
                self.net.cuda()
            self.transform = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            ])

class InceptionIndexer(BaseIndexer):

    def __init__(self):
        self.name = "tfinception"
        self.net = None
        self.tf = True
        self.session = None
        self.graph_def = None
        self.indexed_dirs = set()
        self.index, self.files, self.findex = None, {}, 0

    def load(self):
        if self.session is None:
            logging.warning("Loading the network {}".format(self.name))
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.session = tf.InteractiveSession(config=config)
            network_path = os.path.abspath(__file__).split('indexer.py')[0]+'data/network.pb'
            with gfile.FastGFile(network_path, 'rb') as f:
                self.graph_def = tf.GraphDef()
                self.graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(self.graph_def, name='incept')
                # if png:
                #     png_data = tf.placeholder(tf.string, shape=[])
                #     decoded_png = tf.image.decode_png(png_data, channels=3)
                #     _ = tf.import_graph_def(graph_def, name='incept',input_map={'DecodeJpeg': decoded_png})
                #     return png_data



    def apply(self,image_path):
        self.load()
        if image_path.endswith('.png'):
            im = PIL.Image.open(image_path)
            bg = PIL.Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, im)
            image_path = image_path.replace('.png','.jpg')
            bg.save(image_path)
        pool3 = self.session.graph.get_tensor_by_name('incept/pool_3:0')
        pool3_features = self.session.run(pool3,{'incept/DecodeJpeg/contents:0': file(image_path).read()})
        return np.atleast_2d(np.squeeze(pool3_features))


INDEXERS = {
    'alex':AlexnetIndexer(),
    'inception':InceptionIndexer(),
}

