import mxnet as mx
import numpy as np
import sys, os
from mxnet.io import DataIter
from mxnet.io import DataBatch
import random
import cv2
import helpers
import time

np.random.seed(1301)
random.seed(1301)


class FileIter(DataIter):
    def __init__(self, root_dir, flist_name,
                 regress_overlay=True,
                 cut_off_size=None,
                 data_name="data",
                 label_name="softmax_label",
                 batch_size=1,
                 augment=False,
                 mean_image=None,
                 crop_size=0,
                 random_crop=False,
                 shuffle=False,
                 scale_size=None,
                 crop_indent_x=None,
                 crop_indent_y=None):
        self.regress_overlay = regress_overlay
        self.file_lines = []
        self.epoch = 0
        self.scale_size = scale_size
        self.shuffle = shuffle
        self.label_files = []
        self.image_files = []
        super(FileIter, self).__init__()
        self.batch_size = batch_size
        self.Augment = augment
        self.random = random.Random()
        self.random.seed(1301)
        self.root_dir = root_dir
        self.flist_name = os.path.join(self.root_dir, flist_name)
        self.mean = cv2.imread(mean_image, cv2.IMREAD_GRAYSCALE)
        self.cut_off_size = cut_off_size
        self.data_name = data_name
        self.label_name = label_name
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.crop_indent_x = crop_indent_x
        self.crop_indent_y = crop_indent_y

        self.num_data = len(open(self.flist_name, 'r').readlines())
        #self.num_data = 100
        self.cursor = -1
        self.read_lines()
        self.data, self.label = self._read()
        self.reset()

    def _read(self):
        """get two list, each list contains two elements: name and nd.array value"""
        data = {}
        label = {}

        dd = []
        ll = []
        for i in range(0, self.batch_size):
            line = self.get_line()
            data_img_name, label_img_name = line.strip('\n').split("\t")
            d, l = self._read_img(data_img_name, label_img_name)
            dd.append(d)
            ll.append(l)

        d = np.vstack(dd)
        l = np.vstack(ll)
        data[self.data_name] = d
        if not self.regress_overlay:
            l = l.reshape(l.shape[0])
        label[self.label_name] = l

        res = list(data.items()), list(label.items())
        return res

    def _read_img(self, img_name, label_name):
        img_path = os.path.join(self.root_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(float)  # Image.open(img_path).convert("L")
        if self.regress_overlay:
            label_path = os.path.join(self.root_dir, label_name)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE).astype(float)  # Image.open(label_path).convert("L")
        else:
            label_path = label_name
            label = float(label_name)

        if self.scale_size is not None:
            img = cv2.resize(img, (self.scale_size, self.scale_size), interpolation=cv2.INTER_AREA).astype(float)
            if self.regress_overlay:
                label = cv2.resize(label, (self.scale_size, self.scale_size), interpolation=cv2.INTER_AREA).astype(float)
            # img.thumbnail((self.scale_size, self.scale_size), Image.ANTIALIAS)
            #label.thumbnail((self.scale_size, self.scale_size), Image.ANTIALIAS)

        self.image_files.append(img_path)
        self.label_files.append(label_path)

        if not self.regress_overlay:
            img = np.array(img, dtype=np.float32)  # (h, w, c)
            img = img - self.mean
        # label = np.array(label, dtype=np.float32)  # (h, w)

        if self.Augment:
            rnd_val = self.random.randint(0, 100)
            if rnd_val > 10:
                #for i in range(0, 20):
                #    helpers.ELASTIC_INDICES = None
                #    x = helpers.elastic_transform(img, 150, 15)
                #    cv2.imwrite("c:\\tmp\\img_post"+ str(i) + ".png", x)

                #img = helpers.elastic_transform(img, 50, 10)  # 128
                #label = helpers.elastic_transform(label, 50, 10)

                img = helpers.elastic_transform(img, 150, 15)  # 150, 15 extreme for 256 x 256
                if self.regress_overlay:
                    label = helpers.elastic_transform(label, 150, 15)
                #if img_name == "0001_00000sax_01_09889_IM-4569-0001.png" or True:
                #    time_str = str(str(time.time()).replace(".", ""))
                #    cv2.imwrite("c:\\tmp\\img" + time_str + "_i.png", img)
                # if self.regress_overlay:
                #     cv2.imwrite("c:\\tmp\\img" + time_str + "_l.png", label)

        img = img.reshape(img.shape[0], img.shape[1], 1)

        img /= 256.
        if self.regress_overlay:
            label /= 256.
        else:
            label /= 30.

        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)  # (c, h, w)
        label = np.array(label)  # (h, w)

        if self.crop_size != 0:
            crop_max = img.shape[1] - self.crop_size
            crop_x = crop_max / 2
            crop_y = crop_max / 2
            if self.crop_indent_x is not None:
                crop_x = self.crop_indent_x
            if self.crop_indent_y is not None:
                crop_y = self.crop_indent_y

            if self.random_crop:
                crop_x = self.random.randint(0, crop_max)
                crop_y = self.random.randint(0, crop_max)

            img = img[:, crop_y:crop_y + self.crop_size, crop_x: crop_x + self.crop_size]
            if self.regress_overlay:
                label = label[crop_y:crop_y + self.crop_size, crop_x: crop_x + self.crop_size]

        img = np.expand_dims(img, axis=0)  # (1, c, h, w) or (1, h, w)
        if self.regress_overlay:
            label = label.reshape(1, label.shape[0] * label.shape[1])

        return img, label

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        res = [(k, tuple(list(v.shape[0:]))) for k, v in self.data]
        # print "data : " + str(res)
        return res

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        res = [(k, tuple(list(v.shape[0:]))) for k, v in self.label]
        print "label : " + str(res)
        return res

    def reset(self):
        self.cursor = -1
        self.read_lines()
        helpers.ELASTIC_INDICES = None
        self.label_files = []
        self.image_files = []
        self.epoch += 1

    def getpad(self):
        return 0

    def read_lines(self):
        self.current_line_no = -1;
        with open(self.flist_name, 'r') as f:
            self.file_lines = f.readlines()
            if self.shuffle:
                self.random.shuffle(self.file_lines)

    def get_line(self):
        self.current_line_no += 1
        return self.file_lines[self.current_line_no]


    def iter_next(self):
        self.cursor += self.batch_size
        if self.cursor < self.num_data:
            return True
        else:
            return False

    def eof(self):
        res = self.cursor >= self.num_data
        return res

    def next(self):
        """return one dict which contains "data" and "label" """
        if self.iter_next():
            self.data, self.label = self._read()
            #for i in range(0, 10):
            #    self.data, self.label = self._read()
            #    d.append(mx.nd.array(self.data[0][1]))
            #    l.append(mx.nd.array(self.label[0][1]))
            
            res = DataBatch(data=[mx.nd.array(self.data[0][1])], label=[mx.nd.array(self.label[0][1])], pad=self.getpad(), index=None)
            #if self.cursor % 100 == 0:
            #    print "cursor: " + str(self.cursor)
            return res
        else:
            raise StopIteration