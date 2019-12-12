import os
import scipy.misc
import numpy as np

import ops
import config
import trainer

def main():
    cfg = config.Config(filename_queue="dataset/flowers.tfrecords",
                        logdir="log_flowers")
    t = trainer.Trainer(cfg)

    if not os.path.exists(cfg.sampledir):
        os.makedirs(cfg.sampledir)

    _, im = t.sample(20)
    for i in range(20):
        imname = os.path.join(cfg.sampledir, str(i+1)+".jpg")
        scipy.misc.imsave(imname, im[i])

if __name__ == "__main__":
    main()
