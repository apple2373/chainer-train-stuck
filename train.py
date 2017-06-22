# import sys
# sys.path += ['/mnt/sakura201/stsutsui/anadonda2/lib/python27.zip', '/mnt/sakura201/stsutsui/anadonda2/lib/python2.7', '/mnt/sakura201/stsutsui/anadonda2/lib/python2.7/plat-linux2', '/mnt/sakura201/stsutsui/anadonda2/lib/python2.7/lib-tk', '/mnt/sakura201/stsutsui/anadonda2/lib/python2.7/lib-old', '/mnt/sakura201/stsutsui/anadonda2/lib/python2.7/lib-dynload', '/mnt/sakura201/stsutsui/anadonda2/lib/python2.7/site-packages', '/mnt/sakura201/stsutsui/anadonda2/lib/python2.7/site-packages/Sphinx-1.5.1-py2.7.egg', '/mnt/sakura201/stsutsui/anadonda2/lib/python2.7/site-packages/cupy-1.0.0.1-py2.7-linux-x86_64.egg', '/mnt/sakura201/stsutsui/anadonda2/lib/python2.7/site-packages/keras_contrib-1.2.1-py2.7.egg', '/mnt/sakura201/stsutsui/anadonda2/lib/python2.7/site-packages/Keras-2.0.4-py2.7.egg', '/mnt/sakura201/stsutsui/anadonda2/lib/python2.7/site-packages/Theano-0.9.0-py2.7.egg', '/mnt/sakura201/stsutsui/anadonda2/lib/python2.7/site-packages/chainer-2.0.0-py2.7.egg', '/mnt/sakura201/stsutsui/anadonda2/lib/python2.7/site-packages']

import os

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass

import argparse

import chainer
import numpy as np

from chainer import iterators
from chainer import optimizers
from chainer import training
from chainer.training import extensions

import chainercv
from chainercv.datasets import TransformDataset
from chainercv.links import SegNetBasic

from pixelwise_softmax_classifier  import PixelwiseSoftmaxClassifier
import cv2 as cv

import glob

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--batchsize', type=int, default=6)
parser.add_argument('--out', type=str, default='result')
parser.add_argument('--mode', type=int, required=True, help="0:stuck; 1:works ok with lower res; 2:works ok with serial iterator")
args = parser.parse_args()


class SegDataset(chainer.dataset.DatasetMixin):

    def __init__(self, img_dir, label_dir, ext='png'):
      self.img_fns = glob.glob('{}/*.{}'.format(img_dir, ext))
      self.label_fns = dict([(os.path.basename(fn), fn) for fn in glob.glob('{}/*.{}'.format(label_dir, ext))])

    def get_example(self, i):
      img_fn = self.img_fns[i]
      label_fn = self.label_fns[os.path.basename(img_fn)]
      img = cv.imread(img_fn)[...,::-1]#make it into RGB
      label = cv.imread(label_fn, cv.IMREAD_GRAYSCALE)
      return img,label

    def __len__(self):
        return len(self.img_fns)

# Triggers
log_trigger = (1, 'iteration')
validation_trigger = (10, 'iteration')
end_trigger = (100, 'iteration')

#Dataset Transform
#shape is RGB + 
mean = np.array([0, 0, 0], dtype=np.float32)
std = np.array([1, 1, 1], dtype=np.float32)
width = 1024
height = 512
if args.mode == 1:
  width = 400
  height = 200
def transform(in_data):
  img, label = in_data
  img = cv.resize(img,(width,height))
  img = img.astype(np.float32).transpose(2, 0, 1)#HWC to CHW
  img -= mean[:, None, None]
  img /= std[:, None, None]
  label = label.astype(np.uint8)
  label = cv.resize(label,(width,height))
  label = label.astype(np.int32)
  return  img,label

train = SegDataset(img_dir="./img/",label_dir="./label/")
train = TransformDataset(train,transform)

# Iterator
train_iter = iterators.MultiprocessIterator(train, args.batchsize,n_processes = 8)

if args.mode == 2:
  train_iter = iterators.SerialIterator(train, args.batchsize)

# Model
model = SegNetBasic(n_class=2)
model = PixelwiseSoftmaxClassifier(model)
if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
    model.to_gpu()  # Copy the model to the GPU

# Optimizer
optimizer = optimizers.MomentumSGD(lr=0.1, momentum=0.9)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))

# Updater
updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)

# Trainer
trainer = training.Trainer(updater, end_trigger, out=args.out)
trainer.extend(extensions.LogReport(trigger=log_trigger))
trainer.extend(extensions.snapshot_object(
    model.predictor, filename='model_iteration-{.updater.iteration}'),
    trigger=end_trigger)
trainer.extend(extensions.PrintReport(
    ['epoch', 'iteration', 'elapsed_time',
     'main/loss','main/accuracy']),
    trigger=log_trigger)
trainer.extend(extensions.ProgressBar(update_interval=1))
trainer.run()