from abc import (
  ABC,
  abstractmethod
)
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
import pathlib


random.seed(42)


PCT_VALID = 0.2
CONFIG = BDDConfig('cfg/kastria-local.json')
CLASSIFIER_THRESH = 8000
IMAGENET_STATS = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
KINETICS_STATS = ([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
FRAME_SIZE = (int(720/2), int(1280/2))
CLASS_WEIGHTS = [2.1, 1]
CLASS_WEIGHTS_COMBINED = [1.05, 1]
FRAMES_PER_VIDEO = 32
VIDEO_SIDE_SIZE = 256
VIDEO_CROP_SIZE = 256
CARLA_OFFSET_MS = 3000
SLOWFAST_ALPHA = 4

parser = ArgumentParser()
parser = pytorch_lightning.Trainer.add_argparse_args(parser)
parser.add_argument('--perform-extract', dest='perform_extract', action='store_true', help='Perform the extract process from the original source videos')
parser.add_argument('--arch', dest='arch', action='store', help='Model archetecture to use (default densenet121)')
parser.add_argument('--dense-optical-flow', dest='dense_optical_flow', action='store_true', help='When extracting images, also extract the dense optical flow')
parser.add_argument('--sparse-optical-flow', dest='sparse_optical_flow', action='store_true', help='When extracting images, also extract the sparse optical flow')
parser.add_argument('--perform-carla-mods', dest='perform_carla_mods', action='store_true', help='Perform image modifications (contrast, lighting, blur on the carla images extracted')
parser.add_argument('--perform-stop-start-extract', dest='perform_stop_start_extract', action='store_true', help='Extract the data into the stop start form for motion detection')

parser.add_argument('--single-frame-train', dest='single_frame_train', action='store_true', help='Perform the training process on a single image')

parser.add_argument('--multi-frame-train', dest='multi_frame_train', action='store_true', help='Perform the training process on a multi-frame images')

parser.add_argument('--start-stop-train', dest='start_stop_train', action='store_true', help='Train on the start-stop data')

parser.add_argument('--video-train', dest='video_train', action='store_true', help='Perform the training process on a video clip')

parser.add_argument('--use-bdd-and-carla', dest='bdd_and_carla', action='store_true', help='Perform the training process on a multi-frame images')
parser.add_argument('--carla', dest='carla', action='store', help='Percentage of carla videos to use 1 = 100pct default 1')
parser.add_argument('--bdd', dest='bdd', action='store', help='Percentage of BDD videos to use 1 = 100pct default 0')
parser.add_argument('--oversample-training', dest='oversample_training', action='store_true', help='Oversample the training dataset')

parser.set_defaults(perform_extract = False, single_frame_train = False, multi_frame_train = False, video_train = False, perform_stop_start_extract = False, start_stop_train = False, bdd = 0, carla=1, perform_carla_mods = False, dense_optical_flow = False, sparse_optical_flow = False, oversample_training = False, arch='densenet121')

args = parser.parse_args()

if args.arch not in ['densenet121', 'resnet50']:
  print('Invalid architecture should be [densenet121 | resnet50]')
  exit()

PERFORM_EXTRACT = args.perform_extract
PERFORM_SINGLE_FRAME_TRAIN = args.single_frame_train
PERFORM_MULTI_FRAME_TRAIN = args.multi_frame_train
PERFORM_VIDEO_TRAIN = args.video_train
PERFORM_START_STOP_TRAIN = args.start_stop_train
BDD_AND_CARLA = float(args.bdd) > 0
CARLA_PCT = float(args.carla)


class CarlaImageMod(ABC):
  @abstractmethod
  def filter(self, image):
    pass

class CarlaBlur(CarlaImageMod):
  def __init__(self):
    self.run_prob = 0.3

  def filter(self, image):
    if random.random() <= self.run_prob:
      kernel = numpy.ones((5,5),numpy.float32)/25
      return cv2.filter2D(image,-1,kernel)
    return image

class CarlaBrightness(CarlaImageMod):
  def __init__(self):
    self.run_prob = 0.3

  def filter(self, image):
    if random.random() <= self.run_prob:
      brightness = numpy.random.randint(-50, 50)
      return numpy.clip(image + brightness, 0, 255).astype('uint8')
    return image

class CarlaContrast(CarlaImageMod):
  def __init__(self):
    self.run_prob = 0.3
    self.min_alpha = 0.6
    self.max_alpha = 1.4

  def filter(self, image):
    if random.random() <= self.run_prob:
      alpha_change = (self.max_alpha - self.min_alpha) * numpy.random.random() + self.min_alpha
      return numpy.clip(alpha_change * image, 0, 255).astype('uint8')
    return image

CARLA_IMAGE_MODS = [CarlaBlur(), CarlaBrightness(), CarlaContrast()]

def carla_filter(image):
  for c_filter in CARLA_IMAGE_MODS:
    image = c_filter.filter(image)
  return image

sql = 'SELECT stop_id, carla_id, stop_time_ms, start_time_ms FROM carla_stop WHERE (start_time_ms - stop_time_ms) > 1000 and stop_time_ms > 8000 ORDER BY carla_id'

train_videos = []
valid_videos = []
current_videos = []
carla_videos = {}

with psycopg2.connect(CONFIG.get_psycopg2_conn()) as db:
  with db.cursor() as cursor:
    cursor.execute(sql)

    row = cursor.fetchone()
    movement_tracker = None
    movement_tracker_id = -1
    while row is not None:
      stop_id = row[0]
      carla_id = row[1]
      stop_time_ms = row[2] - 3000
      start_time_ms = row[3] - 3000
      duration = start_time_ms - stop_time_ms
      duration_class = 0
      if duration > CLASSIFIER_THRESH:
        duration_class = 1

      video = {
        'stop_id': stop_id,
        'carla_id': carla_id,
        'stop_time': stop_time_ms / 1000.0,
        'start_time': start_time_ms / 1000.0,
        'stop_time_ms': stop_time_ms,
        'start_time_ms': start_time_ms,
        'duration': duration,
        'duration_class': duration_class,
        'type': 'carla'
      }
      if random.random() <= CARLA_PCT:
        if random.random() <= PCT_VALID:
          valid_videos.append(video)
        else: 
          train_videos.append(video)

      if not str(carla_id) in carla_videos:
        vid = {'carla_id': carla_id, 'stops': []}
        carla_videos[str(carla_id)] = vid
      vid = carla_videos[str(carla_id)]
      stop_array = vid['stops']
      stop_array.append(video)
      row = cursor.fetchone()

def write_video(file_name, frames, fps):
  height, width, layers = frames[0].shape
  vid_size = (width, height)
  out = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'DIVX'), float(fps), vid_size)
  for frame in frames:
    out.write(frame)
  out.release()

if PERFORM_EXTRACT:
  all_videos = valid_videos + train_videos
  for video in all_videos:
    video_file = CONFIG.get_temp_dir() + '/carla-orig/' + str(video['carla_id']) + '.mp4'

    still_dir = CONFIG.get_temp_dir() + '/carla-still/' + str(video['stop_id']) + '-' + str(video['stop_time'])
    multi_still_dir = CONFIG.get_temp_dir() + '/carla-multi-still/' + str(video['stop_id']) + '-' + str(video['stop_time'])
    short_video_file = CONFIG.get_temp_dir() + '/carla-video/' + str(video['stop_id']) + '-' + str(video['stop_time']) + '.mp4'
    dense_optical_flow_still_dir = CONFIG.get_temp_dir() + '/carla-dense-optical-flow/' + str(video['stop_id']) + '-' + str(video['stop_time'])
    dense_optical_flow_video_file = CONFIG.get_temp_dir() + '/carla-dense-optical-flow/' + str(video['stop_id']) + '-' + str(video['stop_time']) + '.mp4'
    sparse_optical_flow_still_dir = CONFIG.get_temp_dir() + '/carla-sparse-optical-flow/' + str(video['stop_id']) + '-' + str(video['stop_time'])
    sparse_optical_flow_video_file = CONFIG.get_temp_dir() + '/carla-sparse-optical-flow/' + str(video['stop_id']) + '-' + str(video['stop_time']) + '.mp4'
    still_dir_path = pathlib.Path(still_dir)
    multi_still_dir_path = pathlib.Path(multi_still_dir)
    short_video_file_path = pathlib.Path(short_video_file)
    dense_optical_flow_still_dir_path = pathlib.Path(dense_optical_flow_still_dir)
    dense_optical_flow_video_file_path = pathlib.Path(dense_optical_flow_video_file)
    sparse_optical_flow_still_dir_path = pathlib.Path(sparse_optical_flow_still_dir)
    sparse_optical_flow_video_file_path = pathlib.Path(sparse_optical_flow_video_file)
    process = False
    if not still_dir_path.exists():
      still_dir_path.mkdir()
      process = True

    if not multi_still_dir_path.exists():
      multi_still_dir_path.mkdir()
      process = True

    if not short_video_file_path.exists():
      process = True

    if args.dense_optical_flow:
      if not dense_optical_flow_still_dir_path.exists():
        dense_optical_flow_still_dir_path.mkdir()
        process = True
      if not dense_optical_flow_video_file_path.exists():
        process = True
      
    if args.sparse_optical_flow:
      if not sparse_optical_flow_still_dir_path.exists():
        sparse_optical_flow_still_dir_path.mkdir()
        process = True
      if not sparse_optical_flow_video_file_path.exists():
        process = True


    for index in range(20):
      still_file_path = pathlib.Path(f'{still_dir}/{str(index)}.jpeg')
      multi_still_file_path = pathlib.Path(f'{multi_still_dir}/{str(index)}.jpeg')
      if (not still_file_path.exists()) or (not multi_still_file_path.exists()):
        process = True


    if process:
      print(f'Processing video {video_file}')
      movement_tracker = DashcamMovementTracker()
      movement_tracker.get_video_frames_from_file(video_file)
      output = movement_tracker.get_training_data(video['stop_time_ms'], args.dense_optical_flow, args.sparse_optical_flow)
      if output is not None:
        stills = output['stills']
        multi_stills = output['multi-stills']
        for index in range(len(stills)):
          output_image_name = still_dir + '/' + str(index) + '.jpeg'
          cv2.imwrite(output_image_name, stills[index])

          output_image_name = multi_still_dir + '/' + str(index) + '.jpeg'
          cv2.imwrite(output_image_name, multi_stills[index])

          if args.dense_optical_flow:
            optical_flow_stills = output['dense-optical-flow-stills']
            output_image_name = dense_optical_flow_still_dir + '/' + str(index) + '.jpeg'
            cv2.imwrite(output_image_name, optical_flow_stills[index])
          
          if args.sparse_optical_flow:
            optical_flow_stills = output['sparse-optical-flow-stills']
            output_image_name = sparse_optical_flow_still_dir + '/' + str(index) + '.jpeg'
            cv2.imwrite(output_image_name, optical_flow_stills[index])
        
        if args.dense_optical_flow:
          write_video(dense_optical_flow_video_file, output['dense-optical-flow-video'], movement_tracker.fps)
        if args.sparse_optical_flow:
          write_video(sparse_optical_flow_video_file, output['sparse-optical-flow-video'], movement_tracker.fps)
        movement_tracker.write_video(short_video_file)

      

if BDD_AND_CARLA:
  bdd_train, bdd_valid = video_stops_from_database(CONFIG)

  print(f'Loading BDD data [{len(bdd_train)}] training videos, [{len(bdd_valid)}] validation videos')

  all_valid = []

  all_train = train_videos + valid_videos
  for video in bdd_train:
    if video['long_stop']:
      video['duration_class'] = 1
    else:
      video['duration_class'] = 0
    all_train.append(video)
  for video in bdd_valid:
    if video['long_stop']:
      video['duration_class'] = 1
    else:
      video['duration_class'] = 0
    all_valid.append(video)

  train_videos = all_train
  valid_videos = all_valid

if args.oversample_training:
  all_train = []
  for video in train_videos:
    if (video['duration'] < 4000 or video['duration'] > 12000):
      all_train.append(video)
      if (video['duration'] < 2000 or video['duration'] > 14000):
        all_train.append(video)
    all_train.append(video)

  train_videos = all_train


print(f'Training Videos = [{len(train_videos)}] validation videos = [{len(valid_videos)}]')


class ImageModel(pytorch_lightning.LightningModule):
  def __init__(self, name = None, trainer = None):
    super(ImageModel, self).__init__()
    if args.arch == 'resnet50':
      self.model = torchvision.models.resnet50(pretrained=True)
      self.model.fc = torch.nn.Linear(in_features=2048, out_features=2)
    else:
      self.model = torchvision.models.densenet121(pretrained=True)
      self.model.classifier = torch.nn.Linear(in_features=1024, out_features=2)

    self.val_confusion = torchmetrics.ConfusionMatrix(num_classes=2)
    if BDD_AND_CARLA:
      self.loss_weights = torch.FloatTensor(CLASS_WEIGHTS_COMBINED).cuda()
    else:
      self.loss_weights = torch.FloatTensor(CLASS_WEIGHTS).cuda()

    self.train_acc = torchmetrics.Accuracy()
    self.valid_acc = torchmetrics.Accuracy()
    self.best_valid_acc = None
    self.name = name
    self.trainer = trainer

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
    pyplot.figure()
    seaborn.set(font_scale=1.2)
    seaborn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d')
    buf = io.BytesIO()
    pyplot.savefig(buf, format='jpeg')
    buf.seek(0)
    im = Image.open(buf)
    im = torchvision.transforms.ToTensor()(im)
    tb.add_image("val_confusion_matrix", im, global_step=self.current_epoch)
    self.val_confusion.reset()
    #self.val_confusion = torchmetrics.ConfusionMatrix(num_classes=2).cuda()
    computed_valid_acc = self.valid_acc.compute()
    total_loss = (sum(output['loss'] for output in outputs)).item()
    self.log('valid_acc', computed_valid_acc)
    if self.best_valid_acc is None:
      self.best_valid_acc = 0
      self.best_valid_loss = 9999999999999
    else:
      dump_model = False
      if self.best_valid_loss > total_loss and self.current_epoch >= 1:
        self.best_valid_loss = total_loss
        dump_model = True
      if self.best_valid_acc < computed_valid_acc and self.current_epoch >= 1:
        self.best_valid_acc = computed_valid_acc
        dump_model = True
      if dump_model and self.trainer is not None:
        trainer.save_checkpoint(f'models/{self.name}-e{self.current_epoch}-a{self.best_valid_acc}.ckpt')

    self.valid_acc.reset()


class ImageDataset(Dataset):
  def __init__(self, training, single_image):
    self.carla_path_prefix = 'carla-still'
    self.bdd_path_prefix = 'bdd-still'
    self.training = training
    if not single_image:
      self.carla_path_prefix = 'carla-multi-still'
      self.bdd_path_prefix = 'bdd-multi-still'

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
    image_file_index = 19
    if self.training:
      image_file_index = random.randint(0, 19)
    image_file = None
    if video['type'] == 'carla':
      image_file = CONFIG.get_temp_dir() + '/' + self.carla_path_prefix + '/' + str(video['stop_id']) + '-' + str(video['stop_time']) + '/' + str(image_file_index) + '.jpeg'
    else:
      image_file = CONFIG.get_temp_dir() + '/' + self.bdd_path_prefix + '/' + video['file_name'] + '-' + str(video['stop_time']) + '/' + str(image_file_index) + '.jpeg'
    image = Image.open(image_file)

    return self.transforms(image), video['duration_class'] 

class ImageDataModule(pytorch_lightning.LightningDataModule):
  def __init__(self, single_image = True):
    super().__init__()
    self.NUM_WORKERS = 4
    self.BATCH_SIZE = 12
    if args.arch == 'resnet50':
      self.BATCH_SIZE = 18
    self.single_image = single_image

  def prepare_data(self):
    print('prepare_data')

  def train_dataloader(self):
    return DataLoader(ImageDataset(True, self.single_image), batch_size=self.BATCH_SIZE, shuffle=True, num_workers=self.NUM_WORKERS)

  def val_dataloader(self):
    return DataLoader(ImageDataset(False, self.single_image), batch_size=self.BATCH_SIZE, shuffle=False, num_workers=self.NUM_WORKERS)


class StartStopImageDataset(Dataset):
  def __init__(self, training, moving_files, stopped_files):
    self.images = []
    self.classes = []

    for path in moving_files:
      self.images.append(path)
      self.classes.append(0)

    for path in stopped_files:
      self.images.append(path)
      self.classes.append(1)

    if training:
      
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
      self.transforms = torchvision.transforms.Compose(
        [
          torchvision.transforms.Resize(FRAME_SIZE),
          torchvision.transforms.ToTensor(),
          torchvision.transforms.Normalize(*IMAGENET_STATS)
        ]
      )

  def __len__(self):
    return len(self.images)

  def __getitem__(self, ix):
    image = Image.open(self.images[ix])
    return self.transforms(image), self.classes[ix]


class StartStopImageDataModule(pytorch_lightning.LightningDataModule):
  def __init__(self):
    super().__init__()
    self.NUM_WORKERS = 4
    self.BATCH_SIZE = 12
    self.train_moving = []
    self.train_stopped = []
    self.valid_moving = []
    self.valid_stopped = []

    base_dir = CONFIG.get_temp_dir() + '/carla-movement/'
    moving = pathlib.Path(base_dir + 'moving/')
    stopped = pathlib.Path(base_dir + 'stopped/')

    for file in moving.iterdir():
      if random.random() > 0.8:
        self.valid_moving.append(file)
      else:
        self.train_moving.append(file)

    for file in stopped.iterdir():
      if random.random() > 0.8:
        self.valid_stopped.append(file)
      else:
        self.train_stopped.append(file)

  def prepare_data(self):
    print('prepare_data')

  def train_dataloader(self):
    return DataLoader(StartStopImageDataset(True, self.train_moving, self.train_stopped), batch_size=self.BATCH_SIZE, shuffle=True, num_workers=self.NUM_WORKERS)

  def val_dataloader(self):
    return DataLoader(StartStopImageDataset(False, self.valid_moving, self.valid_stopped), batch_size=self.BATCH_SIZE * 2, shuffle=False, num_workers=self.NUM_WORKERS)


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

class VideoModel(pytorch_lightning.LightningModule):
  def __init__(self, name = None, trainer = None):
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
    self.train_acc = torchmetrics.Accuracy()
    self.valid_acc = torchmetrics.Accuracy()
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
    loss = torch.nn.functional.cross_entropy(y_hat, y)
    loss = loss.to(torch.float32)
    return loss

  def training_step(self, train_batch, batch_idx):
    y_hat = self.forward(train_batch["video"])
    y = train_batch["label"]
    loss = self.loss_function(y_hat, y)
    batch_value = self.train_acc(y_hat, y)
    self.log('train_loss', loss)
    return loss

  def training_epoch_end(self, outputs):
    self.log('train_acc', self.train_acc.compute())
    self.train_acc.reset()

  def validation_step(self, val_batch, batch_idx):
    video = val_batch['video']
    y_hat = self.forward(video)
    y = val_batch['label']
    loss = self.loss_function(y_hat, y)
    self.val_confusion.update(y_hat, y)
    self.valid_acc.update(y_hat, y)
    self.log('val_loss', loss)
    return { 'loss': loss, 'preds': y_hat, 'target': y}

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
    self.val_confusion.reset()
    #self.val_confusion = torchmetrics.ConfusionMatrix(num_classes=2).cuda()
    computed_valid_acc = self.valid_acc.compute()
    total_loss = (sum(output['loss'] for output in outputs)).item()
    self.log('valid_acc', computed_valid_acc)
    if self.best_valid_acc is None:
      self.best_valid_acc = 0
      self.best_valid_loss = 9999999999999
    else:
      dump_model = False
      if self.best_valid_loss > total_loss and self.current_epoch >= 1:
        self.best_valid_loss = total_loss
        dump_model = True
      if self.best_valid_acc < computed_valid_acc and self.current_epoch >= 1:
        self.best_valid_acc = computed_valid_acc
        dump_model = True
      if dump_model and self.trainer is not None:
        trainer.save_checkpoint(f'models/{self.name}-e{self.current_epoch}-a{self.best_valid_acc}.ckpt')

    self.valid_acc.reset()


def create_video_dataset(videos: typing.List[dict], transforms: typing.Optional[typing.Callable[[dict], typing.Any]]):
  labeled_video_paths = []

  for video in videos:
    labels = {"label": video['duration_class']}
    video_file = CONFIG.get_temp_dir()
    if video['type'] == 'carla':
      video_file = video_file + '/carla-video/' + str(video['stop_id']) + '-' + str(video['stop_time']) + '.mp4'
    else:
      video_file = video_file + '/bdd-video/' + video['file_name'] + '-' + str(video['stop_time']) + '.mp4'
    labels['file_name'] = video_file
    labeled_video_paths.append((video_file, labels))


  return pytorchvideo.data.LabeledVideoDataset(labeled_video_paths, 
    pytorchvideo.data.make_clip_sampler("random", 2),
    transform = transforms)


class VideoDataModule(pytorch_lightning.LightningDataModule):
  def __init__(self):
    super().__init__()
    self.NUM_WORKERS = 8
    self.BATCH_SIZE = 6

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
  data_module = ImageDataModule(True)
  trainer = pytorch_lightning.Trainer.from_argparse_args(args)
  model = ImageModel(name='single-frame', trainer=trainer)
  model.cuda()
  trainer.fit(model, data_module)

if PERFORM_MULTI_FRAME_TRAIN:
  data_module = ImageDataModule(False)
  trainer = pytorch_lightning.Trainer.from_argparse_args(args)
  model = ImageModel(name='multi-frame', trainer=trainer)
  model.cuda()
  trainer.fit(model, data_module)

if PERFORM_START_STOP_TRAIN:
  
  data_module = StartStopImageDataModule()
  trainer = pytorch_lightning.Trainer.from_argparse_args(args)
  model = ImageModel(name='start-stop', trainer=trainer)
  model.cuda()
  trainer.fit(model, data_module)

if PERFORM_VIDEO_TRAIN:
  data_module = VideoDataModule()
  trainer = pytorch_lightning.Trainer.from_argparse_args(args)
  model = VideoModel(name='video', trainer=trainer)
  model.cuda()
  trainer.fit(model, data_module)
