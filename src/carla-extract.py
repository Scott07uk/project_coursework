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
import typing
import pytorchvideo
import pytorchvideo.transforms
import torchvision.transforms._transforms_video
import pytorchvideo.data


random.seed(40)


PCT_VALID = 0.2
CONFIG = BDDConfig('cfg/kastria-local.json')
CLASSIFIER_THRESH = 8000
IMAGENET_STATS = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
KINETICS_STATS = ([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
FRAME_SIZE = (int(720/2), int(1280/2))
CLASS_WEIGHTS = [2, 1]
FRAMES_PER_VIDEO = 8
VIDEO_SIDE_SIZE = 256
VIDEO_CROP_SIZE = 256

parser = ArgumentParser()
parser = pytorch_lightning.Trainer.add_argparse_args(parser)
parser.add_argument('--perform-extract', dest='perform_extract', action='store_true', help='Perform the extract process from the original source videos')

parser.add_argument('--single-frame-train', dest='single_frame_train', action='store_true', help='Perform the training process on a single image')

parser.add_argument('--multi-frame-train', dest='multi_frame_train', action='store_true', help='Perform the training process on a multi-frame images')

parser.add_argument('--video-train', dest='video_train', action='store_true', help='Perform the training process on a video clip')

parser.set_defaults(perform_extract = False, single_frame_train = False, multi_frame_train = False, video_train = False)

args = parser.parse_args()

PERFORM_EXTRACT = args.perform_extract
PERFORM_SINGLE_FRAME_TRAIN = args.single_frame_train
PERFORM_MULTI_FRAME_TRAIN = args.multi_frame_train
PERFORM_VIDEO_TRAIN = args.video_train

sql = 'SELECT stop_id, carla_id, stop_time_ms, start_time_ms FROM carla_stop WHERE (start_time_ms - stop_time_ms) > 1000 and stop_time_ms > 8000 ORDER BY carla_id'

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

        stop_video.cut(start_time = stop_time_ms - 8000, end_time = stop_time_ms - 2000)
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
    self.loss_weights = torch.FloatTensor(CLASS_WEIGHTS).cuda()

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
        frames.shape[1]
      ).long()
    )

    return [slow, fast]

class VideoModel(pytorch_lightning.LightningModule):
  def __init__(self):
    super(VideoModel, self).__init__()
    self.model = model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
    self.model._modules['blocks'][6] = pytorchvideo.models.head.ResNetBasicHead(
      dropout = torch.nn.Dropout(), 
      proj = torch.nn.Linear(in_features=2304, out_features=1),
      output_pool = torch.nn.Identity()
    )
    self.final = torch.nn.Linear(in_features=4, out_features=2)
    self.val_confusion = torchmetrics.ConfusionMatrix(num_classes=2)
    self.loss_weights = torch.FloatTensor(CLASS_WEIGHTS).cuda()

  def forward(self, x):
    out = self.model(x)
    out = self.final(out)
    return out

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=3e-3)
    return optimizer

  def loss_function(self, y_hat, y):
    loss = F.cross_entropy(y_hat, y)
    loss = loss.to(torch.float32)
    return loss

  def training_step(self, train_batch, batch_idx):
    y_hat = self.forward(train_batch["video"])
    loss = self.loss_function(y_hat, train_batch["label"])
    self.log('train_loss', loss)
    return loss

  def validation_step(self, val_batch, batch_idx):
    video = torch.tensor(val_batch['video'])
    print(type(video))
    y_hat = self.forward(video)
    y = val_batch['label']
    loss = self.loss_function(y_hat, y)
    self.val_confusion.update(y_hat, y)
    self.log('val_loss', loss)

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


def create_video_dataset(videos: typing.List[dict], transforms: typing.Optional[typing.Callable[[dict], typing.Any]]):
  labeled_video_paths = []

  for video in videos:
    labels = {"label": video['duration_class']}
    video_file = CONFIG.get_temp_dir() + '/carla-video/' + str(video['stop_id']) + '.mp4'
    labeled_video_paths.append((video_file, labels))


  return pytorchvideo.data.LabeledVideoDataset(labeled_video_paths, 
    pytorchvideo.data.make_clip_sampler("random", 6),
    transform = transforms)


class VideoDataModule(pytorch_lightning.LightningDataModule):
  def __init__(self):
    super().__init__()
    self.NUM_WORKERS = 11
    self.BATCH_SIZE = 1

    self.train_transforms = torchvision.transforms.Compose(
      [
        pytorchvideo.transforms.ApplyTransformToKey(
          key = "video",
          transform = torchvision.transforms.Compose(
            [
              pytorchvideo.transforms.UniformTemporalSubsample(FRAMES_PER_VIDEO),
              torchvision.transforms.Lambda(lambda x: x / 255.0),
              torchvision.transforms._transforms_video.NormalizeVideo(*KINETICS_STATS),
              pytorchvideo.transforms.ShortSideScale(size=VIDEO_SIDE_SIZE),
              torchvision.transforms._transforms_video.CenterCropVideo(VIDEO_CROP_SIZE),
              PackPathway()
            ]
          ),
        ),
      ]
    )

    self.valid_transforms = torchvision.transforms.Compose(
      [
        pytorchvideo.transforms.ApplyTransformToKey(
          key="video",
          transform = torchvision.transforms.Compose(
            [
              pytorchvideo.transforms.UniformTemporalSubsample(FRAMES_PER_VIDEO),
              torchvision.transforms.Lambda(lambda x: x / 255.0),
              torchvision.transforms._transforms_video.NormalizeVideo(*KINETICS_STATS),
              pytorchvideo.transforms.ShortSideScale(size=VIDEO_SIDE_SIZE),
              torchvision.transforms._transforms_video.CenterCropVideo(VIDEO_CROP_SIZE),
              PackPathway()
            ]
          ),
        ),
      ]
    )

  def prepare_data(self):
    print('prepare_data')

  def train_dataloader(self):
    return torch.utils.data.DataLoader(create_video_dataset(train_videos, self.train_transforms), batch_size = self.BATCH_SIZE, num_workers = self.NUM_WORKERS)

  def val_dataloader(self):
    return torch.utils.data.DataLoader(create_video_dataset(valid_videos, self.valid_transforms), batch_size = self.BATCH_SIZE, num_workers = self.NUM_WORKERS)




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

if PERFORM_VIDEO_TRAIN:
  model = VideoModel()
  data_module = VideoDataModule()
  trainer = pytorch_lightning.Trainer.from_argparse_args(args)
  model.cuda()
  trainer.fit(model, data_module)