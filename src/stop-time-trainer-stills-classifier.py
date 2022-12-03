from BDD import (BDDConfig, video_stops_from_database)
from DashcamMovementTracker import DashcamMovementTracker
import psycopg2
import random
import pytorch_lightning
import torchvision.models as models
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pathlib
import cv2
import torch.nn.functional as F
from argparse import ArgumentParser
import torch.nn as nn
from torchvision import transforms
import torch
import os
import time
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
IMAGENET_STATS = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
FRAME_SIZE = (int(720/2), int(1280/2))
DEVICE = 'cuda'
#BATCH_SIZE = 16 #Resnet50
#BATCH_SIZE = 12 #Densenet121
BATCH_SIZE = 3 #EfficientNetB7

video_train = []
video_valid = []
seen_file_names = []

video_train, video_valid = video_stops_from_database(CONFIG)
print(f'Samples training {len(video_train)} validation {len(video_valid)}')

def freeze_layers(model, freeze=True):
  '''Frezes all layers on a model, note the model or layer will be passed by reference, so an in-place modifiation will take place'''
  children = list(model.children())
  for child in children: 
    freeze_layers(child, freeze)
  if not children and not isinstance(model, nn.modules.batchnorm.BatchNorm2d):
    for param in model.parameters(): 
      param.requires_grad = not freeze


class DashcamStopTimeModel(pytorch_lightning.LightningModule):
  def __init__(self, name = None, trainer = None):
    super(DashcamStopTimeModel, self).__init__()

    #Resnet50
    #self.model = models.resnet50(pretrained=True)
    #self.model.fc = nn.Linear(in_features=2048, out_features=2)

    #Densenet121
    #self.model = models.densenet121(pretrained=True)
    #self.model.classifier = nn.Linear(in_features=1024, out_features=2)

    #EfficientNetB7
    self.model = models.efficientnet_b7(pretrained=True)
    self.model.classifier[1] = nn.Linear(in_features=2560, out_features=2)
    
    #freeze_layers(self.model)

    self.val_confusion = ConfusionMatrix(num_classes=2)
    self.loss_weights = torch.FloatTensor([2.35, 4.15]).cuda()

    self.train_acc = torchmetrics.Accuracy()
    self.valid_acc = torchmetrics.Accuracy()
    self.best_valid_loss = None
    self.name = name
    self.trainer = trainer

  def forward(self, x):
    out = self.model(x)
    return out

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=3e-3)
    return optimizer

  def loss_function(self, logits, labels):
    return F.cross_entropy(logits, labels, weight=self.loss_weights)

  def training_step(self, train_batch, batch_idx):
    x, y = train_batch
    logits = self.forward(x)
    loss = self.loss_function(logits, y)
    batch_value = self.train_acc(logits, y)
    self.log('train_loss', loss)
    return loss

  def training_epoch_end(self, outputs):
    self.log('train_acc', self.train_acc.compute())
    self.train_acc.reset()

  def validation_step(self, val_batch, batch_idx):
    x, y = val_batch
    logits = self.forward(x)
    loss = self.loss_function(logits, y)
    self.log('val_loss', loss)
    self.val_confusion.update(logits, y)
    self.valid_acc.update(logits, y)
    return { 'loss': loss, 'preds': logits, 'target': y}

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
    total_loss = (sum(output['loss'] for output in outputs)).item()
    if self.best_valid_loss is None:
      self.best_valid_loss = 9999999999999
    else:
      if self.best_valid_loss > total_loss and self.current_epoch >= 1:
        self.best_valid_loss = total_loss
        if self.trainer is not None:
          self.trainer.save_checkpoint(f'models/{self.name}-e{self.current_epoch}-a{self.best_valid_loss}.ckpt')

    self.valid_acc.reset()

class DashcamDataset(Dataset):
  def __init__(self, data, transform):
    self.data = data
    self.prev_frames = 20
    self.transform = transform

  def __len__(self):
    return len(self.data)


  def __getitem_with_index__(self, ix, image_file_index):
    video = self.data[ix]
    video_dir = pathlib.Path(CONFIG.get_temp_dir() + '/bdd-still/' + video['file_name'] + '-' + str(video['stop_time']))
    
    image_file = f'{video_dir}/{image_file_index}.jpeg'
    
    image_class = 0
    if video['long_stop']:
      image_class = 1
    image = Image.open(image_file)
    return self.transform(image).float(), image_class

  def __getitem__(self, ix):
    image_file_index = random.randint(0, self.prev_frames - 1)
    
    return self.__getitem_with_index__(ix, image_file_index)

  def test(self):
    self.__getitem__(0)

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
#exit()

class DashcamStopTimeDataModule(pytorch_lightning.LightningDataModule):
  def __init__(self, training_videos, validation_videos):
    super().__init__()
    self.NUM_WORKERS = 4
    self.BATCH_SIZE = BATCH_SIZE

    self.train_transforms = transforms.Compose(
      [
        transforms.ColorJitter(0.2, 0.2),
        transforms.RandomHorizontalFlip(0.3),
        transforms.Resize(FRAME_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(*IMAGENET_STATS)
      ]
    )

    self.valid_transforms = transforms.Compose(
      [
        transforms.Resize(FRAME_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(*IMAGENET_STATS)
      ]
    )

    self.training_videos = training_videos
    self.validation_videos = validation_videos

  def prepare_data(self):
    print('prepare_data')

  def train_dataloader(self):
    return DataLoader(DashcamDataset(self.training_videos, self.train_transforms), batch_size=self.BATCH_SIZE, shuffle=True, num_workers=self.NUM_WORKERS)

  def val_dataloader(self):
    return DataLoader(DashcamDataset(self.validation_videos, self.valid_transforms), batch_size=self.BATCH_SIZE, shuffle=True, num_workers=self.NUM_WORKERS)


def main(args):
  data_module = DashcamStopTimeDataModule(video_train, video_valid)
  trainer = pytorch_lightning.Trainer.from_argparse_args(args)
  model = DashcamStopTimeModel(name='still', trainer=trainer)
  model.cuda()
  trainer.fit(model, data_module)



if __name__ == "__main__":
  parser = ArgumentParser()
  parser = pytorch_lightning.Trainer.add_argparse_args(parser)
  args = parser.parse_args()

  main(args)
