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
from os.path import (
  exists
)
from pandas import (
  DataFrame
)
from PIL import (
  Image
)
from pytorch_lightning import (
  LightningDataModule,
  LightningModule,
  Trainer
)
from pytorchvideo.data import (
  LabeledVideoDataset
)
from pytorchvideo.models.resnet import (
  create_resnet
)
from pytorchvideo.transforms import (
  ApplyTransformToKey,
  ShortSideScale,
  UniformTemporalSubsample
)
from torch.utils.data import (
  DataLoader,
  Dataset
)
from torch.nn import (
  BatchNorm3d,
  Module,
  ReLU
)
from torchmetrics import (
  ConfusionMatrix
)
from torchvision.transforms._transforms_video import (
  CenterCropVideo,
  NormalizeVideo
)
from typing import (
  Any, Callable, List, Optional
)

import io
import matplotlib.pyplot
import numpy
import pytorchvideo
import seaborn
import torch
import torch.nn.functional
import pytorchvideo.transforms
import torchvision
import torchvision.transforms

CONFIG_FILE = 'cfg/kastria-local.json'
CONFIG = BDDConfig(CONFIG_FILE)
TMP_DIR_SUFFIX = 'three-channel-video'
#TMP_DIR_SUFFIX = 'video-shorts'
TEMP_DIR = CONFIG.get_temp_dir() + '/' + TMP_DIR_SUFFIX + '/'
KINETICS_STATS = ([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
NUM_WORKERS = 6
SIDE_SIZE = 256
SLOWFAST_ALPHA = 4
CROP_SIZE = 256
NUM_FRAMES = 8
CLIP_DURATION = 4
BATCH_SIZE = 16 #Resnet50
BATCH_SIZE = 12 #Densenet121
BATCH_SIZE = 3 #EfficientNetB7

def pre_process_video_list(videos: List[dict]):
  for video in videos:
    output_file_name = TEMP_DIR + video['file_name']
    if not exists(output_file_name):
      pre_process_video(video, output_file_name)
    

def pre_process_video(video: dict, output_file_name: str):
  TARGET_FPS = 4
  VIDEO_DURATION = 10000
  movement_tracker = DashcamMovementTracker()
  movement_tracker.get_video_frames_from_file(CONFIG.get_absoloute_path_of_video(video['file_type'], video['file_name']))

  #step 1 to 3 channel time series
  movement_tracker.to_channel_time()

  movement_tracker.change_frame_rate(TARGET_FPS)
  video_start_time = max(video['stop_time'] - VIDEO_DURATION, 0)
  video_end_time = video['stop_time']
  movement_tracker.cut(video_start_time, video_end_time)
  print(f'Writing file {output_file_name}')
  movement_tracker.write_video(output_file_name)
  

class DashcamVideoDataset(Dataset):
  def __init__(self, videos: List[dict], num_frames: int = 16, transforms: Optional[Callable[[dict], Any]] = None):
    super().__init__()
    self.videos = videos
    self.num_frames = num_frames
    self.frame_transforms = torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Resize((270, 480)),
      torchvision.transforms.Normalize(*KINETICS_STATS)
    ])
    self.video_transforms = torchvision.transforms.Compose([
      pytorchvideo.transforms.Normalize(*KINETICS_STATS)
    ])
    self.video_transforms = None

  def __len__(self):
    return len(self.videos)

  def __getitem__(self, idx):
    video_dict = self.videos[idx]
    video_file = TEMP_DIR + '/' + video_dict['file_name']
    video_class = 0
    if video_dict['long_stop']:
      video_class = 1
    movement_tracker = DashcamMovementTracker()
    movement_tracker.get_video_frames_from_file(video_file)
    frames = []
    
    for index in range(len(movement_tracker.frames) - self.num_frames, len(movement_tracker.frames)):
      frame = movement_tracker.frames[index]
      if self.frame_transforms is not None:
        frame = self.frame_transforms(frame)
      frames.append(frame)

    frames = torch.stack(frames, 1)
    if self.video_transforms is not None:
      frames = self.video_transforms(frames)

    output = {
      'path': video_file,
      'video': frames, #Tensor of Frames
      'video_name': video_dict['file_name'],
      'label': video_class
    }
    return output



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


  dataset = LabeledVideoDataset(labeled_video_paths, 
    pytorchvideo.data.make_clip_sampler("uniform", CLIP_DURATION),
    transform = transforms,
    decode_audio=False)
  #print(dataset.__next__())
  dataset = DashcamVideoDataset(videos, transforms=transforms)
  
  return dataset


class DashcamStopTimeDataModule(LightningDataModule):
  def __init__(self, training_videos, validation_videos):
    super().__init__()
    self.NUM_WORKERS = NUM_WORKERS
    self.BATCH_SIZE = BATCH_SIZE

    self.transforms = torchvision.transforms.Compose(
      [
        ApplyTransformToKey(
          key = "video",
          transform = torchvision.transforms.Compose(
            [
              UniformTemporalSubsample(NUM_FRAMES),
              torchvision.transforms.Lambda(lambda x: x / 255.0),
              NormalizeVideo(*KINETICS_STATS),
              ShortSideScale(size=SIDE_SIZE),
              CenterCropVideo(CROP_SIZE),
              #PackPathway()
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
    return DataLoader(create_dataset(self.training_videos, self.transforms), batch_size = BATCH_SIZE, num_workers = self.NUM_WORKERS, shuffle = True)

  def val_dataloader(self):
    return DataLoader(create_dataset(self.validation_videos, self.transforms), batch_size = BATCH_SIZE, num_workers = self.NUM_WORKERS)


class PackPathway(Module):
  def __init__(self):
    super().__init__()

  def forward(self, frames: torch.Tensor):
    fast = frames
    slow = torch.index_select(frames, 1, torch.linspace(0, frames.shape[1] - 1, frames.shape[1]).long())
    return [slow, fast]


class DashcamStopTimeClassifier(LightningModule):
  def __init__(self):
    super(DashcamStopTimeClassifier, self).__init__()
    #self.model = model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
    #self.model._modules['blocks'][6] = pytorchvideo.models.head.ResNetBasicHead(
    #  dropout = torch.nn.Dropout(), 
    #  proj = torch.nn.Linear(in_features=2304, out_features=2),
    #  output_pool = torch.nn.Identity()
    #)
    self.model = create_resnet(
      input_channel=3, 
      model_depth=101, 
      model_num_class=2, 
      norm=BatchNorm3d,
      activation=ReLU
    )
    self.model = torch.hub.load("facebookresearch/pytorchvideo:main", model='x3d_s', pretrained=True)
    self.final_layer = torch.nn.Linear(in_features = 400, out_features=2)
    print(self.model)
    self.val_confusion = ConfusionMatrix(num_classes=2)
    self.loss_weights = torch.FloatTensor([1, 8]).cuda()

  def forward(self, x: torch.Tensor):
    out = self.model(x)
    out = self.final_layer(out)
    return out

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=3e-3)

  def loss_function(self, y_hat:torch.Tensor, y:torch.Tensor):
    loss = torch.nn.functional.cross_entropy(y_hat, y, weight=self.loss_weights)
    return loss

  def training_step(self, train_batch, batch_idx):
    y_hat = self.forward(train_batch["video"])
    loss = self.loss_function(y_hat, train_batch["label"])
    self.log('train_loss', loss, batch_size=BATCH_SIZE)
    return loss

  def validation_step(self, val_batch, batch_idx):
    val_batch_video = val_batch['video']
    y = val_batch['label']
    #print(val_batch['video_name'])
    y_hat = self.forward(val_batch_video)
    loss = self.loss_function(y_hat, y)
    self.val_confusion.update(y_hat, y)
    self.log('val_loss', loss, batch_size=BATCH_SIZE)

  def validation_epoch_end(self, outputs):
    tb = self.logger.experiment
    conf_mat = self.val_confusion.compute().detach().cpu().numpy().astype(numpy.int64)
    df_cm = DataFrame(
        conf_mat,
        index=numpy.arange(2),
        columns=numpy.arange(2))
    matplotlib.pyplot.figure()
    seaborn.set(font_scale=1.2)
    seaborn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d')
    buf = io.BytesIO()
    matplotlib.pyplot.savefig(buf, format='jpeg')
    buf.seek(0)
    im = Image.open(buf)
    im = torchvision.transforms.ToTensor()(im)
    tb.add_image("val_confusion_matrix", im, global_step=self.current_epoch, batch_size=BATCH_SIZE)
    self.val_confusion = ConfusionMatrix(num_classes=2).cuda()




# main
parser = ArgumentParser()
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()


training_videos, validation_videos = video_stops_from_database(CONFIG)
print(f'Got {len(training_videos)} training videos and {len(validation_videos)} validation videos')
pre_process_video_list(training_videos)
pre_process_video_list(validation_videos)



model = DashcamStopTimeClassifier()
data_module = DashcamStopTimeDataModule(training_videos, validation_videos)
trainer = Trainer.from_argparse_args(args)
model.cuda()
trainer.fit(model, data_module)


