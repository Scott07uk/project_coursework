from BDD import BDDConfig
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

random.seed(42)

CONFIG = BDDConfig('cfg/kastria-local.json')
PCT_VALID = 0.2
IMAGENET_STATS = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
FRAME_SIZE = (int(720/2), int(1280/2))
DEVICE = 'cuda'

BATCH_SIZE = 16 #Resnet50
BATCH_SIZE = 12 #Densenet121
BATCH_SIZE = 3 #EfficientNetB7

video_train = []
video_valid = []
seen_file_names = []

db = psycopg2.connect(CONFIG.get_psycopg2_conn())
cursor = db.cursor()
cursor.execute("SELECT file_name, file_type, stop_time_ms, start_time_ms, start_time_ms - stop_time_ms as duration from video_file inner join video_file_stop on (id = video_file_id) where stop_time_ms > 0 and start_time_ms is not null AND state = 'DONE' AND stop_time_ms < start_time_ms")
row = cursor.fetchone()

while row is not None:
  video = {
    'file_name': row[0],
    'file_type': row[1],
    'stop_time': float(row[2]),
    'start_time': float(row[3]),
    'duration': float(row[4])
  }

  if video['file_name'] not in seen_file_names:
    seen_file_names.append(video['file_name'])
    if random.random() < PCT_VALID:
      video_valid.append(video)
    else:
      video_train.append(video)


  row = cursor.fetchone()

cursor.close()


print(f'Samples training {len(video_train)} validation {len(video_valid)}')

CONFIG.clean_temp()


def freeze_layers(model, freeze=True):
  '''Frezes all layers on a model, note the model or layer will be passed by reference, so an in-place modifiation will take place'''
  children = list(model.children())
  for child in children: 
    freeze_layers(child, freeze)
  if not children and not isinstance(model, nn.modules.batchnorm.BatchNorm2d):
    for param in model.parameters(): 
      param.requires_grad = not freeze


class DashcamStopTimeModel(pytorch_lightning.LightningModule):
  def __init__(self):
    super(DashcamStopTimeModel, self).__init__()

    # Resnet 50
    #self.model = models.resnet50(pretrained=True) 
    #self.model.fc = nn.Linear(in_features=2048, out_features=1)

    #Densenet121
    #self.model = models.densenet121(pretrained=True) 
    #self.model.classifier = nn.Linear(in_features=1024, out_features=1)

    #EfficientNetB7
    self.model = models.efficientnet_b7(pretrained=True)
    self.model.classifier[1] = nn.Linear(in_features=2560, out_features=1)
    #freeze_layers(self.model)

  def forward(self, x):
    out = self.model(x)
    return out

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=3e-3)
    return optimizer

  def loss_function(self, logits, labels):
    return F.l1_loss(logits, labels)
    #return F.mse_loss(logits, labels)

  def training_step(self, train_batch, batch_idx):
    x, y = train_batch
    logits = self.forward(x)
    loss = self.loss_function(logits, y)
    self.log('train_loss', loss)
    return loss

  def validation_step(self, val_batch, batch_idx):
    x, y = val_batch
    logits = self.forward(x)
    loss = self.loss_function(logits, y)
    self.log('val_loss', loss)

    

class DashcamDataset(Dataset):
  def __init__(self, data, transform):
    self.data = data
    self.prev_frames = 20
    self.transform = transform

  def __len__(self):
    return len(self.data)

  def video_from_frames(self, ix):
    video = self.data[ix]
    video_file_name = video['file_name']

    cursor = db.cursor()
    cursor.execute("SELECT file_name FROM video_file WHERE file_name = '" + video_file_name + "' FOR UPDATE")

    video_dir = pathlib.Path(CONFIG.get_temp_dir() + '/single-image/' + video_file_name)
    video_stop_time = video['stop_time']
    if not video_dir.exists():
      video_dir.mkdir()
    movement_tracker = DashcamMovementTracker()
    times, frames = movement_tracker.get_video_frames_from_file(CONFIG.get_absoloute_path_of_video(video['file_type'], video_file_name))
    if times is None:
      print('times is none')
    print(f'Extracted {len(times)} times and {len(frames)} from {video_file_name}')
    video_files_written = 0
    for index in range(len(times) - self.prev_frames):
      #print(f'Checking to see if {times[index + self.prev_frames]} >= {video_stop_time} ({video_file_name})')
      if times[index + self.prev_frames] >= video_stop_time:
        for use_index in range(self.prev_frames):
          output_image_name = f'{video_dir}/{use_index}.jpeg'
          print(f'Writing {output_image_name}')
          cv2.imwrite(output_image_name, frames[index])
          video_files_written += 1
        break

    print(f'Video files written {video_files_written}')
    cursor.close()

  def __getitem__(self, ix):
    video = self.data[ix]
    video_dir = pathlib.Path(CONFIG.get_temp_dir() + '/single-image/' + video['file_name'])
    
    image_file_index = random.randint(0, self.prev_frames - 1)
    image_file = f'{video_dir}/{image_file_index}.jpeg'
    #print(f'Does file exist {image_file}')
    while not os.path.exists(image_file):
      self.video_from_frames(ix)
      time.sleep(1)
    
    try:
      image = Image.open(image_file)
      return self.transform(image).float(), video['duration']
    except:
      print(f'Exception thrown reading file {image_file}')

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
  model = DashcamStopTimeModel()
  data_module = DashcamStopTimeDataModule(video_train, video_valid)
  trainer = pytorch_lightning.Trainer.from_argparse_args(args)
  model.cuda()
  trainer.fit(model, data_module)



if __name__ == "__main__":
  parser = ArgumentParser()
  parser = pytorch_lightning.Trainer.add_argparse_args(parser)
  args = parser.parse_args()

  main(args)
