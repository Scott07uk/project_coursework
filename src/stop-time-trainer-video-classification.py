from BDD import (BDDConfig, video_stops_from_database)
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
from torchvision import transforms
import torch
import os
import time
import numpy
import pytorchvideo.models.resnet
from os.path import exists
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    UniformTemporalSubsample,
    RandomShortSideScale,
    Normalize
)
from typing import Any, Callable, List, Optional
import torchmetrics
import io
import numpy
import pandas
import matplotlib.pyplot as plt
import seaborn as sns

#https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09+

random.seed(42)

CLASSIFIER_THRESH = 8000
CONFIG = BDDConfig('cfg/kastria-local.json')
PCT_VALID = 0.2
IMAGENET_STATS = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
FRAME_SIZE = (int(720/2), int(1280/2))
DEVICE = 'cuda'
BATCH_SIZE = 2
TEMP_DIR = CONFIG.get_temp_dir() + '/video-shorts/'

video_train = []
video_valid = []
video_all = []
seen_file_names = []

video_train, video_valid = video_stops_from_database(CONFIG)

all_videos = video_train + video_valid


print(f'Samples training {len(video_train)} validation {len(video_valid)}')

CONFIG.clean_temp()

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

class DashcamStopTimeModel(pytorch_lightning.LightningModule):
  def __init__(self, name=None, trainer=None):
    super(DashcamStopTimeModel, self).__init__()
    self.model = pytorchvideo.models.resnet.create_resnet(
      input_channel=3, # RGB input from Kinetics
      model_depth=50, # For the tutorial let's just use a 50 layer network
      model_num_class=2, # Kinetics has 400 classes so we need out final head to align
      norm=nn.BatchNorm3d
    )
    print(self.model)
    #self.model.fc = nn.Linear(in_features=2048, out_features=1)
    #self.model.classifier = nn.Linear(in_features=1024, out_features=1)
    #freeze_layers(self.model)
    self.loss_weights = torch.FloatTensor([2.3, 4.15]).cuda()
    #self.loss_weights = torch.FloatTensor([4.23, 6.81]).cuda()

    self.val_confusion = torchmetrics.ConfusionMatrix(num_classes=2)
    self.train_acc = torchmetrics.Accuracy()
    self.valid_acc = torchmetrics.Accuracy()
    self.best_valid_loss = None
    self.best_valid_acc = None
    self.name = name
    self.trainer = trainer

  def forward(self, x):
    out = self.model(x)
    return out

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=3e-3)
    return optimizer

  def loss_function(self, y_hat, y):
    loss = F.cross_entropy(y_hat, y, weight=self.loss_weights)
    return loss

  def training_step(self, train_batch, batch_idx):
    y = train_batch["label"]
    y_hat = self.model(train_batch["video"])
    loss = self.loss_function(y_hat, y)
    self.log('train_loss', loss)
    batch_value = self.train_acc(y_hat, y)
    return loss

  def training_epoch_end(self, outputs):
    self.log('train_acc', self.train_acc.compute())
    self.train_acc.reset()

  def validation_step(self, val_batch, batch_idx):
    y = val_batch['label']
    y_hat = self.model(val_batch['video'])
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
    im = transforms.ToTensor()(im)
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
    image_class = 0
    if video['long_stop']:
      image_class = 1
    labels = {"label": image_class, "stop_time": video['stop_time']}
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
    self.NUM_WORKERS = 8
    self.BATCH_SIZE = BATCH_SIZE

    self.train_transforms = transforms.Compose(
      [
        ApplyTransformToKey(
          key="video",
          transform=transforms.Compose(
            [
              UniformTemporalSubsample(32),
              transforms.Lambda(lambda x: x / 255.0),
              Normalize(*IMAGENET_STATS),
              RandomShortSideScale(min_size=256, max_size=320),
              transforms.RandomCrop(244),
              transforms.RandomHorizontalFlip(p=0.5),
            ]
          ),
        ),
      ]
    )

    self.valid_transforms = transforms.Compose(
      [
        ApplyTransformToKey(
          key="video",
          transform= transforms.Compose(
            [
              UniformTemporalSubsample(8),
              transforms.Lambda(lambda x: x / 255.0),
              Normalize(*IMAGENET_STATS),
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
  trainer = pytorch_lightning.Trainer.from_argparse_args(args)
  model = DashcamStopTimeModel(name='video', trainer=trainer)
  data_module = DashcamStopTimeDataModule(video_train, video_valid)
  
  model.cuda()
  trainer.fit(model, data_module)



if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument('--extract-video', default='False') 
  parser = pytorch_lightning.Trainer.add_argparse_args(parser)
  args = parser.parse_args()

  main(args)
