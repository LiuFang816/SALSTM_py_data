#!/usr/bin/env python
# encoding: utf-8
# Copyright (c) 2015 Grzegorz Chrupała
# A simple encoder-decoder example with funktional
from __future__ import division 
import theano
import numpy
import random
import itertools
import cPickle as pickle
import argparse
import gzip
import sys
import os
import funktional.util as util
import copy
import time
from funktional.layer import *

class EncoderDecoder(Layer):
    """A simple encoder-decoder net with shared input and output vocabulary."""
    def __init__(self, size_vocab, size, depth):
        self.size_vocab  = size_vocab
        self.size     = size
        self.depth    = depth
        self.Embed    = Embedding(self.size_vocab, self.size)
        encoder = lambda size_in, size: StackedGRUH0(size_in, size, self.depth)
        decoder = lambda size_in, size: StackedGRU(size_in, size, self.depth)
        self.Encdec   = EncoderDecoderGRU(self.size, self.size, self.size, 
                                          encoder=encoder,
                                          decoder=decoder)
        self.Out      = Dense(size_in=self.size, size_out=self.size)
        self.params   = self.Embed.params + self.Encdec.params + self.Out.params

    def __call__(self, inp, out_prev):
        return softmax3d(self.Embed.unembed(self.Out(self.Encdec(self.Embed(inp), self.Embed(out_prev)))))

class Model(object):
    """Trainable encoder-decoder model."""
    def __init__(self, size_vocab, size, depth):
        self.size = size
        self.size_vocab = size_vocab
        self.depth = depth
        self.network = EncoderDecoder(self.size_vocab, self.size, self.depth)
        self.input       = T.imatrix()
        self.output_prev = T.imatrix()
        self.output      = T.imatrix()
        self.projection  = last(self.network.Encdec.Encode(self.network.Embed(self.input)))
        OH = OneHot(size_in=self.size_vocab)
        self.output_oh   = OH(self.output)
        self.output_pred = self.network(self.input, self.output_prev)
        self.cost = CrossEntropy(self.output_oh, self.output_pred)
        self.updater = Adam()
        self.updates = self.updater.get_updates(self.network.params, self.cost)
        self.train = theano.function([self.input, self.output_prev, self.output ], 
                                      self.cost, updates=self.updates)
        self.predict = theano.function([self.input, self.output_prev], self.output_pred)
        self.project = theano.function([self.input], self.projection)
        # Like train, but no updates
        self.loss = theano.function([self.input, self.output_prev, self.output ], self.cost)
        
        
        
def pad(xss, padding):
    max_len = max((len(xs) for xs in xss))
    def pad_one(xs):
        return xs + [ padding for _ in range(0,(max_len-len(xs))) ]
    return [ pad_one(xs) for xs in xss ]

def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF G
    args = [iter(iterable)] * n
    return itertools.izip(*args)

def batch(item, BEG, END):
    """Prepare minibatch."""
    mb = numpy.array(pad([[BEG]+s+[END] for s in item], END), dtype='int32')
    inp = mb[:,1:]
    out = mb[:,1:]
    out_prev = mb[:,0:-1]
    return (inp, out_prev, out)

def batch_para(item, BEG, END):
    """Prepare minibatch."""
    mb_inp = numpy.array(pad([[BEG]+s+[END] for s,_ in item], END), dtype='int32')
    mb_out = numpy.array(pad([[BEG]+r+[END] for _,r in item], END), dtype='int32')
    inp = mb_inp[:,1:]
    out = mb_out[:,1:]
    out_prev = mb_out[:,0:-1]
    return (inp, out_prev, out)

def valid_loss(model, inp, out, BEG, END, batch_size=128):
    costs = 0.0; N = 0
    for _j, item in enumerate(grouper(itertools.izip(inp, out), batch_size)):
        j = _j + 1
        inp, out_prev, out = batch_para(item, BEG, END)
        costs = costs + model.loss(inp, out_prev, out) ; N = N + 1
    return costs / N

def shuffled(x):
    y = copy.copy(x)
    random.shuffle(y)
    return y

def main():
    parser = argparse.ArgumentParser(description='Stacked recurrent autoencoder for sentences.')
    subparsers = parser.add_subparsers(title='Commands',
                                       dest='command',
                                       description='Valid commands',
                                       help='Additional help')

    parser_train = subparsers.add_parser('train', help='Train a new model from data')
    parser_train.add_argument('--size',   type=int, default=512,       help='Size of embeddings and hidden layers')
    parser_train.add_argument('--depth',  type=int, default=2,         help='Number of hidden layers')
    parser_train.add_argument('--epochs', type=int, default=1,         help='Number of training epochs')
    parser_train.add_argument('--batch_size', type=int, default=128,   help='Number of examples in minibatch')
    parser_train.add_argument('--seed',   type=int, default=None,      help='Random seed')
    parser_train.add_argument('--log',    type=str, default='log.txt', help='Path to log file')
    parser_train.add_argument('--model_path', type=str, default='.',       help='Path to model directory')
    parser_train.add_argument('train_file',    type=str,                    help='Path to training data')
    parser_train.add_argument('valid_file',    type=str,                    help='Path to validation data')
    parser_train.add_argument('--train_file_out', type=str, default=None,   help='Path to training data output (unless same as input)')
    parser_train.add_argument('--valid_file_out', type=str, default=None,   help='Path to validation data output (unless same as input')
    parser_proj = subparsers.add_parser('encode', help='Encode data using trained model')
    parser_proj.add_argument('model_path',     type=str,               help='Path to model')
    parser_proj.add_argument('input_file',     type=str,               help='Path to data')
    parser_proj.add_argument('output_file',    type=str,               help='Path to output data')
    args = parser.parse_args()
    if args.command == 'train':
        train_cmd(args)
    elif args.command == 'encode':
        encode_cmd(args)

def train_cmd(args):
    if args.seed is not None:
        random.seed(args.seed)
    mapper = util.IdMapper(min_df=10)
    text_in      = ( line.split() for line in open(args.train_file) )
    text_out     = text_in if args.train_file_out is None else ( line.split() for line in open(args.train_file_out) )
    text_val_in  = (line.split() for line in open(args.valid_file))
    text_val_out = text_val_in if args.valid_file_out is None else (line.split() for line in open(args.valid_file_out))
    sents_in      = mapper.fit_transform(text_in)
    sents_out     = mapper.transform(text_out)
    sents_val_in  = list(mapper.transform(text_val_in))
    sents_val_out = list(mapper.transform(text_val_out))
    sents = shuffled(list(itertools.izip(sents_in, sents_out)))
    pickle.dump(mapper, gzip.open(os.path.join(args.model_path, 'mapper.pkl.gz'),'w'))
    mb_size = 128
    model = Model(size_vocab=mapper.size(), size=args.size, depth=args.depth)
    with open(args.log,'w') as log:
        for epoch in range(1,args.epochs + 1):
            costs = 0 ; N = 0
            for _j, item in enumerate(grouper(sents, args.batch_size)):
                j = _j + 1
                inp, out_prev, out = batch_para(item, mapper.BEG_ID, mapper.END_ID)
                costs = costs + try_alloc(lambda: model.train(inp, out_prev, out), attempts=5, pause=10) ; N = N + 1
                print epoch, j, "train", costs / N
                if j % 500 == 0:
                    cost_valid = valid_loss(model, sents_val_in, sents_val_out, mapper.BEG_ID, mapper.END_ID, batch_size=args.batch_size)
                    print epoch, j, "valid", cost_valid
                if j % 100 == 0:
                    pred = model.predict(inp, out_prev)
                    for i in range(len(pred)):
                        orig = [ w for w in list(mapper.inverse_transform([inp[i]]))[0] 
                                 if w != mapper.END ]
                        res =  [ w for w in list(mapper.inverse_transform([numpy.argmax(pred, axis=2)[i]]))[0] 
                                 if w != mapper.END ]
                        log.write("{}".format(' '.join(orig)))
                        log.write("\n")
                        log.write("{}".format(' '.join(res)))
                        log.write("\n")
                    log.flush()
            pickle.dump(model, gzip.open(os.path.join(args.model_path,'model.{0}.pkl.gz'.format(epoch)),'w'))
    pickle.dump(model, gzip.open(os.path.join(args.model_path, 'model.pkl.gz'), 'w'))

def try_alloc(fn, attempts=1, pause=1):
    '''Try executing function `fn`, recovering from MemoryError `attempts` times.'''
    try:
        return fn()
    except MemoryError as e:
        if attempts == 0:
            raise e
        else:
            sys.stderr.write("MemoryError: Trying again {0} times\n".format(attempts))
            time.sleep(pause)
            return try_alloc(fn, attempts=attempts-1, pause=pause*2)
        
def encode(model, mapper, sents):
    """Return projections of `sents` to the final hidden state of the encoder of `model`."""
    return numpy.vstack([  model.project(batch(item, mapper.BEG_ID, mapper.END_ID)[0]) 
                           for item in grouper(mapper.transform(sents), 128) ])
def encode_cmd(args):
    model = pickle.load(gzip.open(os.path.join(args.model_path, 'model.pkl.gz')))
    mapper = pickle.load(gzip.open(os.path.join(args.model_path, 'mapper.pkl.gz')))
    sents = [line.split() for line in open(args.input_file) ]
    pickle.dump(encode(model, mapper, sents), gzip.open(args.output_file, 'w'))

if __name__ == '__main__':
    main()
