import pprint
import cPickle
import numpy as np

pp = pprint.PrettyPrinter()

def save_pkl(path, obj):
  with open(path, 'w') as f:
    cPickle.dump(obj, f)
    print(" [*] save %s" % path)

def load_pkl(path):
  with open(path) as f:
    obj = cPickle.load(f)
    print(" [*] load %s" % path)
    return obj

def save_npy(path, obj):
  np.save(path, obj)
  print(" [*] save %s" % path)

def load_npy(path):
  obj = np.load(path)
  print(" [*] load %s" % path)
  return obj
