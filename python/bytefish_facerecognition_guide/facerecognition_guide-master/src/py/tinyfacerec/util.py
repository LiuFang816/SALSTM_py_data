import os, sys
import numpy as np
import PIL.Image as Image

def normalize(X, low, high, dtype=None):
	X = np.asarray(X)
	minX, maxX = np.min(X), np.max(X)
	# normalize to [0...1].	
	X = X - float(minX)
	X = X / float((maxX - minX))
	# scale to [low...high].
	X = X * (high-low)
	X = X + low
	if dtype is None:
		return np.asarray(X)
	return np.asarray(X, dtype=dtype)

def read_images(path, sz=None):
	c = 0
	X,y = [], []
	for dirname, dirnames, filenames in os.walk(path):
		for subdirname in dirnames:
			subject_path = os.path.join(dirname, subdirname)
			for filename in os.listdir(subject_path):
				try:
					im = Image.open(os.path.join(subject_path, filename))
					im = im.convert("L")
					# resize to given size (if given)
					if (sz is not None):
						im = im.resize(sz, Image.ANTIALIAS)
					X.append(np.asarray(im, dtype=np.uint8))
					y.append(c)
				except IOError:
					print "I/O error({0}): {1}".format(errno, strerror)
				except:
					print "Unexpected error:", sys.exc_info()[0]
					raise
			c = c+1
	return [X,y]

def asRowMatrix(X):
	if len(X) == 0:
		return np.array([])
	mat = np.empty((0, X[0].size), dtype=X[0].dtype)
	for row in X:
		mat = np.vstack((mat, np.asarray(row).reshape(1,-1)))
	return mat

def asColumnMatrix(X):
	if len(X) == 0:
		return np.array([])
	mat = np.empty((X[0].size, 0), dtype=X[0].dtype)
	for col in X:
		mat = np.hstack((mat, np.asarray(col).reshape(-1,1)))
	return mat
