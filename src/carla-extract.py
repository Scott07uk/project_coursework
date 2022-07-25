from argparse import (
  ArgumentParser
)
from BDD import (
  BDDConfig
)
from DashcamMovementTracker import (
  DashcamMovementTracker
)
from PIL import (
  Image
)
from torch.utils.data import ( 
  Dataset,
  DataLoader
)
from matplotlib import (
  pyplot
)
import psycopg2
import cv2
import numpy
import random
import pytorch_lightning
import torchvision
import torch
import torchmetrics
import pandas
import seaborn
import io

random.seed(40)

PERFORM_EXTRACT = False
PERFORM_SINGLE_FRAME_TRAIN = False
PERFORM_MULTI_FRAME_TRAIN = True
PCT_VALID = 0.2
CONFIG = BDDConfig('cfg/kastria-local.json')
CLASSIFIER_THRESH = 8000
IMAGENET_STATS = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
FRAME_SIZE = (int(720/2), int(1280/2))

parser = ArgumentParser()
parser = pytorch_lightning.Trainer.add_argparse_args(parser)
args = parser.parse_args()

sql = 'SELECT stop_id, carla_id, stop_time_ms, start_time_ms FROM carla_stop WHERE (start_time_ms - stop_time_ms) > 1000 and stop_time_ms > 4000 ORDER BY carla_id'

train_videos = []
valid_videos = []

with psycopg2.connect(CONFIG.get_psycopg2_conn()) as db:
  with db.cursor() as cursor:
    cursor.execute(sql)

    row = cursor.fetchone()
    movement_tracker = None
    movement_tracker_id = -1
    while row is not None:
      stop_id = row[0]
      carla_id = row[1]
      stop_time_ms = row[2]
      start_time_ms = row[3]
      duration = start_time_ms - stop_time_ms
      duration_class = 0
      if duration > CLASSIFIER_THRESH:
        duration_class = 1

      video = {
        'stop_id': stop_id,
        'carla_id': carla_id,
        'stop_time_ms': stop_time_ms,
        'start_time_ms': start_time_ms,
        'duration': duration,
        'duration_class': duration_class
      }

      if random.random() <= PCT_VALID:
        valid_videos.append(video)
      else: 
        train_videos.append(video)
      
      if PERFORM_EXTRACT:
        orig_file_name = CONFIG.get_temp_dir() + '/carla-orig/' + str(carla_id) + '.mp4'
        if carla_id != movement_tracker_id:
          movement_tracker = DashcamMovementTracker()
          movement_tracker.get_video_frames_from_file(orig_file_name)
          movement_tracker_id = carla_id

        stop_video = DashcamMovementTracker()
        stop_video.fps = movement_tracker.fps
        stop_video.frame_stop_status = movement_tracker.frame_stop_status.copy()
        stop_video.frame_times = movement_tracker.frame_times.copy()
        stop_video.frames = movement_tracker.frames.copy()

        stop_video.cut(start_time = stop_time_ms - 6000, end_time = stop_time_ms - 2000)
        #video
        stop_video.write_video(CONFIG.get_temp_dir() + '/carla-video/' + str(stop_id) + '.mp4')

        #Single Still at stop time
        cv2.imwrite(CONFIG.get_temp_dir() + '/carla-still/' + str(stop_id) + '.jpeg', stop_video.frames[-1])
 
        #Multi-still
        red = cv2.cvtColor(stop_video.frames[0], cv2.COLOR_BGR2GRAY)
        green = cv2.cvtColor(stop_video.frames[int(len(stop_video.frames) / 2)], cv2.COLOR_BGR2GRAY)
        blue = cv2.cvtColor(stop_video.frames[-1], cv2.COLOR_BGR2GRAY)
        output_image = numpy.dstack([red, green, blue]).astype(numpy.uint8)
        cv2.imwrite(CONFIG.get_temp_dir() + '/carla-multi-still/' + str(stop_id) + '.jpeg', output_image)

      row = cursor.fetchone()


print(f'Training Videos = [{len(train_videos)}] validation videos = [{len(valid_videos)}]')


class ImageModel(pytorch_lightning.LightningModule):
  def __init__(self):
    super(ImageModel, self).__init__()
    self.model = torchvision.models.densenet121(pretrained=True)
    print(self.model)
    self.model.classifier = torch.nn.Linear(in_features=1024, out_features=2)

    self.val_confusion = torchmetrics.ConfusionMatrix(num_classes=2)
    self.loss_weights = torch.FloatTensor([2.25, 1]).cuda()

  def forward(self, x):
    out = self.model(x)
    return out

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=3e-3)
    return optimizer

  def loss_function(self, logits, labels):
    return torch.nn.functional.cross_entropy(logits, labels, weight=self.loss_weights)

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
    self.val_confusion.update(logits, y)
    return { 'loss': loss, 'preds': logits, 'target': y}

  def validation_epoch_end(self, outputs):
    tb = self.logger.experiment
    conf_mat = self.val_confusion.compute().detach().cpu().numpy().astype(numpy.int64)
    df_cm = pandas.DataFrame(
        conf_mat,
        index=numpy.arange(2),
        columns=numpy.arange(2))
    pyplot.figure()
    seaborn.set(font_scale=1.2)
    seaborn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d')
    buf = io.BytesIO()
    pyplot.savefig(buf, format='jpeg')
    buf.seek(0)
    im = Image.open(buf)
    im = torchvision.transforms.ToTensor()(im)
    tb.add_image("val_confusion_matrix", im, global_step=self.current_epoch)
    self.val_confusion = torchmetrics.ConfusionMatrix(num_classes=2).cuda()

class ImageDataset(Dataset):
  def __init__(self, training, single_image):
    self.path_prefix = 'carla-still'
    if not single_image:
      self.path_prefix = 'carla-multi-still'

    if training:
      self.data = train_videos
      
      self.transforms = torchvision.transforms.Compose(
        [
          torchvision.transforms.ColorJitter(0.2, 0.2),
          torchvision.transforms.RandomHorizontalFlip(0.3),
          torchvision.transforms.Resize(FRAME_SIZE),
          torchvision.transforms.ToTensor(),
          torchvision.transforms.Normalize(*IMAGENET_STATS)
        ]
      )
    else:
      self.data = valid_videos
      self.transforms = torchvision.transforms.Compose(
        [
          torchvision.transforms.Resize(FRAME_SIZE),
          torchvision.transforms.ToTensor(),
          torchvision.transforms.Normalize(*IMAGENET_STATS)
        ]
      )

  def __len__(self):
    return len(self.data)

  def __getitem__(self, ix):
    video = self.data[ix]
    image_file = CONFIG.get_temp_dir() + '/' + self.path_prefix + '/' + str(video['stop_id']) + '.jpeg'
    image = Image.open(image_file)

    return self.transforms(image), video['duration_class'] 

class ImageDataModule(pytorch_lightning.LightningDataModule):
  def __init__(self, single_image = True):
    super().__init__()
    self.NUM_WORKERS = 4
    self.BATCH_SIZE = 12
    self.single_image = single_image

  def prepare_data(self):
    print('prepare_data')

  def train_dataloader(self):
    return DataLoader(ImageDataset(True, self.single_image), batch_size=self.BATCH_SIZE, shuffle=True, num_workers=self.NUM_WORKERS)

  def val_dataloader(self):
    return DataLoader(ImageDataset(False, self.single_image), batch_size=self.BATCH_SIZE, shuffle=False, num_workers=self.NUM_WORKERS)


if PERFORM_SINGLE_FRAME_TRAIN:
  model = ImageModel()
  data_module = ImageDataModule(True)
  trainer = pytorch_lightning.Trainer.from_argparse_args(args)
  model.cuda()
  trainer.fit(model, data_module)

if PERFORM_MULTI_FRAME_TRAIN:
  model = ImageModel()
  data_module = ImageDataModule(False)
  trainer = pytorch_lightning.Trainer.from_argparse_args(args)
  model.cuda()
  trainer.fit(model, data_module)
