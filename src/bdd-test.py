from argparse import (
  ArgumentParser
)
from BDD import (
  BDDConfig, 
  video_stops_from_database
)
import cv2
from DashcamMovementTracker import (DashcamMovementTracker)
import numpy
from PIL import (
  Image
)
import pathlib
import psycopg2
import pytorch_lightning
import torch
import torchvision
import pytorchvideo.data
import pytorchvideo.models
import pytorchvideo.transforms
import os
import time
import torchvision.transforms._transforms_video

IMAGENET_STATS = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
FRAME_SIZE = (int(720/2), int(1280/2))
KINETICS_STATS = ([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
SLOWFAST_ALPHA = 4

class DensenetClassificationImageModel(pytorch_lightning.LightningModule):
  def __init__(self):
    super(DensenetClassificationImageModel, self).__init__()
    self.model = torchvision.models.densenet121()
    self.model.classifier = torch.nn.Linear(in_features=1024, out_features=2)

  def forward(self, x):
    return self.model(x)

class ResnetClassificationImageModel(pytorch_lightning.LightningModule):
  def __init__(self):
    super(ResnetClassificationImageModel, self).__init__()
    self.model = torchvision.models.resnet50()
    self.model.fc = torch.nn.Linear(in_features=2048, out_features=2)

  def forward(self, x):
    return self.model(x)

class EfficientnetClassificationImageModel(pytorch_lightning.LightningModule):
  def __init__(self):
    super(EfficientnetClassificationImageModel, self).__init__()
    self.model = torchvision.models.efficientnet_b7()
    self.model.classifier[1] = torch.nn.Linear(in_features=2560, out_features=2)

  def forward(self, x):
    return self.model(x)

class ResnetClassificationVideoModel(pytorch_lightning.LightningModule):
  def __init__(self):
    super(ResnetClassificationVideoModel, self).__init__()
    self.model = pytorchvideo.models.resnet.create_resnet(
      input_channel=3, # RGB input from Kinetics
      model_depth=50, # For the tutorial let's just use a 50 layer network
      model_num_class=2, # Kinetics has 400 classes so we need out final head to align
      norm=torch.nn.BatchNorm3d
    )

  def forward(self, x):
    return self.model(x)

class SlowFastClassificationVideoModel(pytorch_lightning.LightningModule):
  def __init__(self):
    super(SlowFastClassificationVideoModel, self).__init__()
    self.model = model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=False)
    self.model._modules['blocks'][6] = pytorchvideo.models.head.ResNetBasicHead(
      dropout = torch.nn.Dropout(), 
      proj=torch.nn.Linear(in_features=2304, out_features=1),
      output_pool = torch.nn.Identity()
    )
    self.final = torch.nn.Linear(in_features=4, out_features=2)

  def forward(self, x):
    out = self.model(x)
    return self.final(out)


class ResnetRegressionImageModel(pytorch_lightning.LightningModule):
  def __init__(self):
    super(ResnetRegressionImageModel, self).__init__()
    self.model = torchvision.models.resnet50()
    self.model.fc = torch.nn.Linear(in_features=2048, out_features=1)

  def forward(self, x):
    return self.model(x)

class DensenetRegressionImageModel(pytorch_lightning.LightningModule):
  def __init__(self):
    super(DensenetRegressionImageModel, self).__init__()
    self.model = torchvision.models.densenet121()
    self.model.classifier = torch.nn.Linear(in_features=1024, out_features=1)

  def forward(self, x):
    return self.model(x)

class EfficientnetRegressionImageModel(pytorch_lightning.LightningModule):
  def __init__(self):
    super(EfficientnetRegressionImageModel, self).__init__()
    self.model = torchvision.models.efficientnet_b7()
    self.model.classifier[1] = torch.nn.Linear(in_features=2560, out_features=1)

  def forward(self, x):
    return self.model(x)

class SlowFastRegressionVideoModel(pytorch_lightning.LightningModule):
  def __init__(self):
    super(SlowFastRegressionVideoModel, self).__init__()
    self.model = model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
    self.model._modules['blocks'][6] = pytorchvideo.models.head.ResNetBasicHead(
      dropout = torch.nn.Dropout(), 
      proj=torch.nn.Linear(in_features=2304, out_features=1),
      output_pool = torch.nn.Identity()
    )
    self.final = torch.nn.Linear(in_features=4, out_features=1)

  def forward(self, x):
    out = self.model(x)
    return self.final(out)

class ResnetRegressionVideoModel(pytorch_lightning.LightningModule):
  def __init__(self):
    super(ResnetRegressionVideoModel, self).__init__()
    self.model = pytorchvideo.models.resnet.create_resnet(
      input_channel=3, # RGB input from Kinetics
      model_depth=50, # For the tutorial let's just use a 50 layer network
      model_num_class=1, # Kinetics has 400 classes so we need out final head to align
      norm=torch.nn.BatchNorm3d
    )

  def forward(self, x):
    return self.model(x)


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

transforms = torchvision.transforms.Compose(
  [
    torchvision.transforms.Resize(FRAME_SIZE),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(*IMAGENET_STATS)
  ]
)

resnet_video_transforms = torchvision.transforms.Compose(
  [
    pytorchvideo.transforms.ApplyTransformToKey(
      key="video",
      transform= torchvision.transforms.Compose(
        [
          pytorchvideo.transforms.UniformTemporalSubsample(8),
          torchvision.transforms.Lambda(lambda x: x / 255.0),
          pytorchvideo.transforms.Normalize(*IMAGENET_STATS),
        ]
      ),
    ),
  ]
)

slowfast_video_transforms = torchvision.transforms.Compose(
  [
    pytorchvideo.transforms.ApplyTransformToKey(
      key="video",
      transform = torchvision.transforms.Compose(
        [
          pytorchvideo.transforms.UniformTemporalSubsample(32),
          torchvision.transforms.Lambda(lambda x: x / 255.0),
          torchvision.transforms._transforms_video.NormalizeVideo(*KINETICS_STATS),
          pytorchvideo.transforms.ShortSideScale(size=256),
          torchvision.transforms._transforms_video.CenterCropVideo(256),
          PackPathway()
        ]
      ),
    ),
  ]
)

parser = ArgumentParser()

parser.add_argument('--model', dest='model_file', action='store', help='File name of the model to use')
parser.add_argument('--config', dest='config', action='store', help='The config file to use')
parser.add_argument('--regression', dest='regression', action='store_true', help = 'Test a regression model')
parser.add_argument('--images', dest='images', action='store', help = 'Use the specified image type default multi-still')
parser.add_argument('--arch', dest='arch', action='store', help = 'Network arch to use default resnet50')
parser.add_argument('--csv', dest='csv', action='store_true', help ='Output as CSV')

parser.set_defaults(config = 'cfg/kastria-local.json', regression = False, images='multi-still', arch='resnet50', csv=False)
args = parser.parse_args()

CONFIG = BDDConfig(args.config)

model = None
if args.regression:
  if args.arch == 'resnet50':
    if args.images == 'video':
      model = ResnetRegressionVideoModel.load_from_checkpoint(args.model_file)
    else:
      model = ResnetRegressionImageModel.load_from_checkpoint(args.model_file)
  elif args.arch == 'densenet121':
    model = DensenetRegressionImageModel.load_from_checkpoint(args.model_file)
  elif args.arch == 'efficientnet_b7':
    model = EfficientnetRegressionImageModel.load_from_checkpoint(args.model_file)
  elif args.arch == 'slowfast':
    model = SlowFastRegressionVideoModel.load_from_checkpoint(args.model_file)
  else:
    print(f'{args.arch} is not valid arch, please use resnet50, densenet121, efficientnet_b7, slowfast')
    exit()
else:
  if args.arch == 'resnet50':
    if args.images == 'video':
      model = ResnetClassificationVideoModel.load_from_checkpoint(args.model_file)
    else:
      model = ResnetClassificationImageModel.load_from_checkpoint(args.model_file)
  elif args.arch == 'densenet121':
    model = DensenetClassificationImageModel.load_from_checkpoint(args.model_file)
  elif args.arch == 'efficientnet_b7':
    model = EfficientnetClassificationImageModel.load_from_checkpoint(args.model_file)
  elif args.arch == 'slowfast':
    model = SlowFastClassificationVideoModel.load_from_checkpoint(args.model_file)
  else:
    print(f'{args.arch} is not valid arch, please use resnet50, densenet121, efficientnet_b7, slowfast')
    exit()

model.eval()

def run_inference(image_file_name):
  pil_input_image = Image.open(image_file_name)
  input_tensor = transforms(pil_input_image)
  input_tensor = input_tensor.unsqueeze(0)
  output_tensor = model.forward(input_tensor)
  if args.regression:
    return output_tensor[0].item()
  _, y_hat = output_tensor.max(1)
  return y_hat.item()
  
  if output_tensor[0] > output_tensor[1]:
    return 0
  return 1

video_train, video_test = video_stops_from_database(CONFIG)

correct = [0, 0]
incorrect = [0, 0]
total_error_sec = 0
min_duration = 99999999
max_duration = 0

if args.csv:
  print('FileName,Actual,Predicted,InferenceTime')

start_time = time.time() * 1000

if args.images == 'video':
  
  use_video_transforms = None
  multiplier = 1.0
  if args.arch == 'resnet50':
    use_video_transforms = resnet_video_transforms
    multiplier = 40.0 * 1000.0
  elif args.arch == 'slowfast':
    use_video_transforms = slowfast_video_transforms
  
  labeled_video_paths = []

  for video in video_test:
    labels = {"label": video['duration'], "stop_time": video['stop_time'], "file_name": video['file_name'], "long_stop": video['long_stop'], "duration": video['duration']}
    video_file = CONFIG.get_temp_dir() + '/bdd-video/' + video['file_name'] + '-' + str(video['stop_time']) + '.mp4'
    if os.path.exists(video_file):
      labeled_video_paths.append((video_file, labels))

  video_data_set = pytorchvideo.data.LabeledVideoDataset(labeled_video_paths, 
    pytorchvideo.data.make_clip_sampler("random", 2),
    transform = use_video_transforms)

  dataloader = torch.utils.data.DataLoader(video_data_set, num_workers=0, batch_size=1)
  for video in dataloader:
    vid_start_time = time.time() * 1000
    video_data = video['video']
    label = video['label']
    file_name = video['file_name']
    file_name = file_name[0]
    raw_output = model.forward(video_data)
    vid_run_time = (time.time() * 1000) - vid_start_time
    if args.regression:
      model_output = raw_output[0].item() * multiplier
      min_duration = min(min_duration, model_output)
      max_duration = max(max_duration, model_output)
      error = abs(label - model_output)
      total_error_sec = total_error_sec + error
      if args.csv:
        print(file_name + ',' + str(label.item()) + ',' + str(model_output) + ',' + str(vid_run_time))
      else:
        print('Video [' + file_name + '] stop duration = [' + str(label.item()) + '] pred duration = [' + str(model_output) + ']')
    else:
      _, y_hat = raw_output.max(1)
      model_output = y_hat.item()
      if video['long_stop'].item():
        if model_output == 0:
          incorrect[1] = incorrect[1] + 1
        else:
          correct[1] = correct[1] + 1
      else:
        if model_output == 0:
          correct[0] = correct[0] + 1
        else:
          incorrect[0] = incorrect[0] + 1
      print('Video [' + file_name + '] stop duration = [' + str(video['duration'].item()) + '] long stop = [' + str(video['long_stop'].item()) + '] predicted = [' + str(model_output) + ']')

else:
  for video in video_test:
    vid_start_time = time.time() * 1000
    image_file_name = None
    if args.images == 'still':
      image_file_name = CONFIG.get_temp_dir() + '/bdd-still/' + video['file_name'] + '-' + str(video['stop_time']) + '/19.jpeg'
    elif args.images == 'multi-still':  
      image_file_name = CONFIG.get_temp_dir() + '/bdd-multi-still/' + video['file_name'] + '-' + str(video['stop_time']) + '/19.jpeg'
    model_output = run_inference(image_file_name)
    vid_run_time = (time.time() * 1000) - vid_start_time
    if args.regression:
      min_duration = min(min_duration, model_output)
      max_duration = max(max_duration, model_output)
      error = abs(video['duration'] - model_output)
      total_error_sec = total_error_sec + error
      if args.csv:
        print(video['file_name'] + ',' + str(video['start_time'] - video['stop_time']) + ',' + str(model_output)  + ',' + str(vid_run_time))
      else:
        print('Video [' + video['file_name'] + '] stop duration = [' + str(video['duration']) + '] pred duration = [' + str(model_output) + ']')
    else:
      if video['long_stop']:
        if model_output == 0:
          incorrect[1] = incorrect[1] + 1
        else:
          correct[1] = correct[1] + 1
      else:
        if model_output == 0:
          correct[0] = correct[0] + 1
        else:
          incorrect[0] = incorrect[0] + 1
      print('Video [' + video['file_name'] + '] stop duration = [' + str(video['start_time'] - video['stop_time']) + '] long stop = [' + str(video['long_stop']) + '] predicted = [' + str(model_output) + ']')

end_time = time.time() * 1000

if args.regression:
  print(f'Total videos [{str(len(video_test))}]')
  print(f'Total error [{str(total_error_sec)}]')
  print(f'Mean error [{str(total_error_sec / len(video_test))}]')
  print(f'Max prediction [{str(max_duration)}]')
  print(f'Min prediction [{str(min_duration)}]')
else:
  print(f'Correct: short: {str(correct[0])}, long: {str(correct[1])}')
  print(f'Incorrect: short: {str(incorrect[0])}, long: {str(incorrect[1])}')
  accuracy = sum(correct) / (sum(correct) + sum(incorrect))
  print(f'Accuracy {accuracy}')

print(f'Total Time taken {str(end_time - start_time)} ms')
print(f'Mean Time Per video {str((end_time - start_time) / len(video_test))}')