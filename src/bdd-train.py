from argparse import (
  ArgumentParser
)
from BDD import (
  BDDConfig, 
  video_stops_from_database
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
import os


random.seed(40)


PCT_VALID = 0.2
CLASSIFIER_THRESH = 8000
IMAGENET_STATS = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
KINETICS_STATS = ([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
FRAME_SIZE = (int(720/2), int(1280/2))
CLASS_WEIGHTS = [2.1, 1]
CLASS_WEIGHTS_COMBINED = [1.05, 1]
FRAMES_PER_VIDEO = 8
VIDEO_SIDE_SIZE = 256
VIDEO_CROP_SIZE = 256
VIDEO_DIR = '/bdd-video'
STILL_DIR = '/bdd-still'
MULTI_STILL_DIR = '/bdd-multi-still'

parser = ArgumentParser()
parser = pytorch_lightning.Trainer.add_argparse_args(parser)
parser.add_argument('--perform-extract', dest='perform_extract', action='store_true', help='Perform the extract process from the original source videos')

parser.add_argument('--single-frame-train', dest='single_frame_train', action='store_true', help='Perform the training process on a single image')

parser.add_argument('--multi-frame-train', dest='multi_frame_train', action='store_true', help='Perform the training process on a multi-frame images')

parser.add_argument('--video-train', dest='video_train', action='store_true', help='Perform the training process on a video clip')

parser.add_argument('--config', dest='config', action='store', help='Use the named config file')

parser.set_defaults(perform_extract = False, single_frame_train = False, multi_frame_train = False, video_train = False, config = 'cfg/kastria-local.json')

args = parser.parse_args()

PERFORM_EXTRACT = args.perform_extract
PERFORM_SINGLE_FRAME_TRAIN = args.single_frame_train
PERFORM_MULTI_FRAME_TRAIN = args.multi_frame_train
PERFORM_VIDEO_TRAIN = args.video_train

CONFIG = BDDConfig(args.config)

train_videos, valid_videos = video_stops_from_database(CONFIG, MIN_DURATION=1000)
all_videos = []

for video in train_videos:
  all_videos.append(video)
for video in valid_videos:
  all_videos.append(video)


def mkdir_if_not_exists(dir):
  print(f'Going to make [{dir}] if it does not exist')
  if not os.path.exists(dir):
    os.mkdir(dir)

if PERFORM_EXTRACT:
  VIDEO_DIR = CONFIG.get_temp_dir() + VIDEO_DIR
  mkdir_if_not_exists(VIDEO_DIR)
  VIDEO_DIR = VIDEO_DIR + '/'

  STILL_DIR = CONFIG.get_temp_dir() + STILL_DIR
  mkdir_if_not_exists(STILL_DIR)
  STILL_DIR = STILL_DIR + '/'

  MULTI_STILL_DIR = CONFIG.get_temp_dir() + MULTI_STILL_DIR
  mkdir_if_not_exists(MULTI_STILL_DIR)
  MULTI_STILL_DIR = MULTI_STILL_DIR + '/'
  

  for video in all_videos:
    orig_file_name = CONFIG.get_video_dir('train') + '/' + video['file_name']
    movement_tracker = DashcamMovementTracker()
    movement_tracker.get_video_frames_from_file(orig_file_name)

    stop_video = DashcamMovementTracker()
    stop_video.fps = movement_tracker.fps
    stop_video.frame_stop_status = movement_tracker.frame_stop_status.copy()
    stop_video.frame_times = movement_tracker.frame_times.copy()
    stop_video.frames = movement_tracker.frames.copy()

    stop_video.cut(start_time = video['stop_time'] - 8000, end_time = video['stop_time'] - 2000)
    #video
    output_file = VIDEO_DIR + video['file_name'] + '-' + str(video['stop_time']) + '.mp4'
    stop_video.write_video(output_file)
    
    #Single Still at stop time
    cv2.imwrite(STILL_DIR + video['file_name'] + '-' + str(video['stop_time']) + '.jpeg', stop_video.frames[-1])
 
    #Multi-still
    red = cv2.cvtColor(stop_video.frames[0], cv2.COLOR_BGR2GRAY)
    green = cv2.cvtColor(stop_video.frames[int(len(stop_video.frames) / 2)], cv2.COLOR_BGR2GRAY)
    blue = cv2.cvtColor(stop_video.frames[-1], cv2.COLOR_BGR2GRAY)
    output_image = numpy.dstack([red, green, blue]).astype(numpy.uint8)
    cv2.imwrite(MULTI_STILL_DIR + video['file_name'] + '-' + str(video['stop_time']) + '.jpeg', output_image)
  
print(f'Training Videos = [{len(train_videos)}] validation videos = [{len(valid_videos)}]')


class ImageModel(pytorch_lightning.LightningModule):
  def __init__(self, arch = 'densenet121'):
    super(ImageModel, self).__init__()
    if arch == 'densenet121':
      self.model = torchvision.models.densenet121(pretrained=True)
      self.model.classifier = torch.nn.Linear(in_features=1024, out_features=2)
    elif arch == 'resnet50':
      self.model = torchvision.models.resnet50(pretrained=True)
      self.model.fc = nn.Linear(in_features=2048, out_features=2)
    elif arch == 'efficientnet_b7':
      self.model = models.efficientnet_b7(pretrained=True)
      self.model.classifier[1] = nn.Linear(in_features=2560, out_features=2)

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
    self.path_prefix = STILL_DIR
    if not single_image:
      self.path_prefix = MULTI_STILL_DIR

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
    image = None
    image_file = self.path_prefix + video['file_name'] + '.jpeg'
    image = Image.open(image_file)
    image_class = 0
    if video['long_stop']:
      image_class = 1
    return self.transforms(image), image_class

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
    video_file = VIDEO_DIR + video['file_name'] + '-' + str(video['stop_time']) + '.mp4'
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
