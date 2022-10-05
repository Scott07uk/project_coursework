from BDD import (
  BDDConfig,
  video_stops_from_database
)
from DashcamMovementTracker import DashcamMovementTracker
import psycopg2
import random
import pytorch_lightning
import torchvision.models as models
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pytorchvideo.data
import pathlib
import cv2
import torch.nn.functional as F
from argparse import ArgumentParser
import torch.nn as nn
import torch
import os
import time
import numpy
import pytorchvideo.models.resnet
from os.path import exists
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)
from torchvision.transforms import Compose, Lambda, ToTensor
from torchvision.transforms._transforms_video import (
  CenterCropVideo,
  NormalizeVideo
)
from typing import Any, Callable, List, Optional
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics import ConfusionMatrix
import io
import numpy
import torchmetrics

random.seed(42)

CLASSIFIER_THRESH = 8000
CONFIG = BDDConfig('cfg/kastria-local.json')
PCT_VALID = 0.2
KINETICS_STATS = ([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
FRAME_SIZE = (int(720/2), int(1280/2))
DEVICE = 'cuda'
BATCH_SIZE = 6
TEMP_DIR = CONFIG.get_temp_dir() + '/video-shorts/'
#TEMP_DIR = CONFIG.get_temp_dir() + '/three-channel-video/'
SIDE_SIZE = 256
SLOWFAST_ALPHA = 4
CROP_SIZE = 256
NUM_FRAMES = 32
SAMPLING_RATE = 2
DEFAULT_FPS = 15
DEFAULT_CLIP_DURATION = (NUM_FRAMES * SAMPLING_RATE) / DEFAULT_FPS

video_train = []
video_valid = []
video_all = []
seen_file_names = []

#db = psycopg2.connect(CONFIG.get_psycopg2_conn())
#cursor = db.cursor()
#cursor.execute("SELECT file_name, file_type, stop_time_ms, start_time_ms, (start_time_ms - stop_time_ms) as duration FROM video_file INNER JOIN video_file_stop ON (id = video_file_id) WHERE stop_time_ms > 4000 and start_time_ms is not null AND state = 'DONE' AND stop_time_ms < start_time_ms")
#row = cursor.fetchone()

#while row is not None:
#  video = {
#    'file_name': row[0],
#    'file_type': row[1],
#    'stop_time': float(row[2]),
#    'start_time': float(row[3]),
#    'duration': float(row[4]),
#    'long_stop': row[4] >= CLASSIFIER_THRESH
#  }

#  if video['file_name'] not in seen_file_names:
#    seen_file_names.append(video['file_name'])
#    if random.random() < PCT_VALID:
#      video_valid.append(video)
#    else:
#      video_train.append(video)
#    video_all.append(video)


#  row = cursor.fetchone()

#cursor.close()

video_train, video_valid = video_stops_from_database(CONFIG)


print(f'Samples training {len(video_train)} validation {len(video_valid)}')

#CONFIG.clean_temp()

def raw_video_to_shorts(video_files):
  for file in video_files:
    absoloute_file_name = CONFIG.get_absoloute_path_of_video(file['file_type'], file['file_name'])
    output_file = TEMP_DIR + '/' + file['file_name']
    frame_time_max = file['stop_time']
    frame_time_min = max(frame_time_max - 4500, 0)

    if not exists(output_file):
      movement_tracker = DashcamMovementTracker()
      frame_times, frames = movement_tracker.get_video_frames_from_file(absoloute_file_name)
      new_image_size = (int(frames[0].shape[1] / 2), int(frames[0].shape[0] / 2))
      new_frames = []
      for index in range(len(frames)):
        if frame_times[index] >= frame_time_min and frame_times[index] <= frame_time_max:
          new_frames.append(cv2.resize(frames[index], new_image_size, interpolation = cv2.INTER_AREA))
        else:
          if frame_times[index] > frame_time_max and frames is not None:
            break

      movement_tracker.frames = new_frames
      movement_tracker.write_video(output_file)


class PackPathway(torch.nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, frames: torch.Tensor):
    fast = frames

    slow = torch.index_select(
      frames,
      1,
      torch.linspace(
        0,
        frames.shape[1] - 1,
        frames.shape[1] // SLOWFAST_ALPHA
      ).long()
    )

    return [slow, fast]


class DashcamStopTimeModel(pytorch_lightning.LightningModule):
  def __init__(self, name = None, trainer = None):
    super(DashcamStopTimeModel, self).__init__()
    self.model = model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
    self.model._modules['blocks'][6] = pytorchvideo.models.head.ResNetBasicHead(
      dropout = nn.Dropout(), 
      proj=nn.Linear(in_features=2304, out_features=1),
      output_pool = nn.Identity()
    )
    self.final = nn.Linear(in_features=4, out_features=2)
    
    self.val_confusion = torchmetrics.ConfusionMatrix(num_classes=2)
    self.train_acc = torchmetrics.Accuracy()
    self.valid_acc = torchmetrics.Accuracy()
    self.best_valid_loss = None
    self.best_valid_acc = None
    self.name = name
    self.trainer = trainer

  def forward(self, x):
    out = self.model(x)
    out = self.final(out)
    return out

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=3e-3)
    return optimizer

  def loss_function(self, y_hat, y):
    #y = y.to(torch.float32)

    loss = F.cross_entropy(y_hat, y)
    loss = loss.to(torch.float32)
    #print(loss)
    return loss

  def training_step(self, train_batch, batch_idx):
    y = train_batch["label"]
    y_hat = self.forward(train_batch["video"])
    loss = self.loss_function(y_hat, y)
    self.log('train_loss', loss)
    batch_value = self.train_acc(y_hat, y)
    return loss

  def training_epoch_end(self, outputs):
    self.log('train_acc', self.train_acc.compute())
    self.train_acc.reset()

  def validation_step(self, val_batch, batch_idx):
    y = val_batch['label']
    y_hat = self.forward(val_batch['video'])
    loss = self.loss_function(y_hat, y)
    self.log('val_loss', loss)
    self.val_confusion.update(y_hat, y)
    self.valid_acc.update(y_hat, y)
    return { 'loss': loss, 'preds': y_hat, 'target': y}

  def validation_epoch_end(self, outputs):
    tb = self.logger.experiment
    conf_mat = self.val_confusion.compute().detach().cpu().numpy().astype(numpy.int64)
    df_cm = pandas.DataFrame(
        conf_mat,
        index=numpy.arange(2),
        columns=numpy.arange(2))
    plt.figure()
    sns.set(font_scale=1.2)
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d')
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    im = Image.open(buf)
    im = ToTensor()(im)
    tb.add_image("val_confusion_matrix", im, global_step=self.current_epoch)
    self.val_confusion.reset()
    computed_valid_acc = self.valid_acc.compute()
    self.log('valid_acc', computed_valid_acc)
    computed_valid_acc = computed_valid_acc.item()
    self.valid_acc.reset()
    total_loss = (sum(output['loss'] for output in outputs)).item()
    if self.best_valid_loss is None:
      self.best_valid_loss = 9999999999999
      self.best_valid_acc = 0
    else:
      dump_model = False
      if self.best_valid_loss > total_loss and self.current_epoch >= 1:
        self.best_valid_loss = total_loss
        dump_model = True
      if self.best_valid_acc < computed_valid_acc and self.current_epoch >= 1:
        self.best_valid_acc = computed_valid_acc
        dump_model = True
      if dump_model:
        if self.trainer is not None:
          self.trainer.save_checkpoint(f'models/{self.name}-e{self.current_epoch}-a{self.best_valid_acc}.ckpt')



def create_dataset(videos: List[dict], transforms: Optional[Callable[[dict], Any]]):
  labeled_video_paths = []

  for video in videos:
    video_class = 0
    if video['long_stop']:
      video_class = 1
    labels = {"label": video_class, "stop_time": video['stop_time']}
    video_file = TEMP_DIR + '/' + video['file_name']
    if exists(video_file):
      labeled_video_paths.append((video_file, labels))


  return pytorchvideo.data.LabeledVideoDataset(labeled_video_paths, 
    pytorchvideo.data.make_clip_sampler("random", 2),
    transform = transforms)


#train_transforms = transforms.Compose(
#      [
#        transforms.ColorJitter(0.2, 0.2),
#        transforms.RandomHorizontalFlip(0.3),
#        transforms.Resize(FRAME_SIZE),
#        transforms.ToTensor(),
#        transforms.Normalize(*IMAGENET_STATS)
#      ]
#    )    
#test = DashcamDataset(video_train, train_transforms)
#test.test()
#create_dataset(video_train)
#exit()

class DashcamStopTimeDataModule(pytorch_lightning.LightningDataModule):
  def __init__(self, training_videos, validation_videos):
    super().__init__()
    self.NUM_WORKERS = 11
    self.BATCH_SIZE = BATCH_SIZE

    self.train_transforms = Compose(
      [
        ApplyTransformToKey(
          key = "video",
          transform = Compose(
            [
              UniformTemporalSubsample(NUM_FRAMES),
              Lambda(lambda x: x / 255.0),
              NormalizeVideo(*KINETICS_STATS),
              ShortSideScale(size=SIDE_SIZE),
              CenterCropVideo(CROP_SIZE),
              PackPathway()
            ]
          ),
        ),
      ]
    )

    self.valid_transforms = Compose(
      [
        ApplyTransformToKey(
          key="video",
          transform = Compose(
            [
              UniformTemporalSubsample(NUM_FRAMES),
              Lambda(lambda x: x / 255.0),
              NormalizeVideo(*KINETICS_STATS),
              ShortSideScale(size=SIDE_SIZE),
              CenterCropVideo(CROP_SIZE),
              PackPathway()
            ]
          ),
        ),
      ]
    )

    self.training_videos = training_videos
    self.validation_videos = validation_videos

  def prepare_data(self):
    print('prepare_data')

  def train_dataloader(self):
    return DataLoader(create_dataset(self.training_videos, self.train_transforms), batch_size = BATCH_SIZE, num_workers = self.NUM_WORKERS)

  def val_dataloader(self):
    return DataLoader(create_dataset(self.validation_videos, self.valid_transforms), batch_size = BATCH_SIZE, num_workers = self.NUM_WORKERS)


def main(args):
  #raw_video_to_shorts(video_all)
  #exit()
  data_module = DashcamStopTimeDataModule(video_train, video_valid)
  trainer = pytorch_lightning.Trainer.from_argparse_args(args)
  model = DashcamStopTimeModel(name='video', trainer=trainer)
  model.cuda()
  trainer.fit(model, data_module)



if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument('--extract-video', default='False') 
  parser = pytorch_lightning.Trainer.add_argparse_args(parser)
  args = parser.parse_args()

  main(args)