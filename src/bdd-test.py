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

IMAGENET_STATS = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
FRAME_SIZE = (int(720/2), int(1280/2))

#Model copied from carla-extract
class DensenetClassificationImageModel(pytorch_lightning.LightningModule):
  def __init__(self):
    super(DensenetClassificationImageModel, self).__init__()
    self.model = torchvision.models.densenet121()
    self.model.classifier = torch.nn.Linear(in_features=1024, out_features=2)

  def forward(self, x):
    return self.model(x)

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

transforms = torchvision.transforms.Compose(
  [
    torchvision.transforms.Resize(FRAME_SIZE),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(*IMAGENET_STATS)
  ]
)

parser = ArgumentParser()

parser.add_argument('--model', dest='model_file', action='store', help='File name of the model to use')
parser.add_argument('--config', dest='config', action='store', help='The config file to use')
parser.add_argument('--regression', dest='regression', action='store_true', help = 'Test a regression model')
parser.add_argument('--images', dest='images', action='store', help = 'Use the specified image type default multi-still')
parser.add_argument('--arch', dest='arch', action='store', help = 'Network arch to use default resnet50')

parser.set_defaults(config = 'cfg/kastria-local.json', regression = False, images='multi-still', arch='resnet50')
args = parser.parse_args()

CONFIG = BDDConfig(args.config)

model = None
if args.regression:
  if args.arch == 'resnet50':
    model = ResnetRegressionImageModel.load_from_checkpoint(args.model_file)
  elif args.arch == 'densenet121':
    model = DensenetRegressionImageModel.load_from_checkpoint(args.model_file)
else:
  model = DensenetClassificationImageModel.load_from_checkpoint(args.model_file)

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

for video in video_test:
  image_file_name = None
  if args.images == 'still':
    image_file_name = CONFIG.get_temp_dir() + '/bdd-still/' + video['file_name'] + '-' + str(video['stop_time']) + '.jpeg'
  else:  
    image_file_name = CONFIG.get_temp_dir() + '/bdd-multi-still/' + video['file_name'] + '-' + str(video['stop_time']) + '.jpeg'
  model_output = run_inference(image_file_name)
  if args.regression:
    min_duration = min(min_duration, model_output)
    max_duration = max(max_duration, model_output)
    error = abs(video['duration'] - model_output)
    total_error_sec = total_error_sec + error
    print('Video [' + video['file_name'] + '] stop duration = [' + str(video['start_time'] - video['stop_time']) + '] pred duration = [' + str(model_output) + ']')
  else:
    if video['long_stop']:
      if image_class == 0:
        incorrect[1] = incorrect[1] + 1
      else:
        correct[1] = correct[1] + 1
    else:
      if image_class == 0:
        correct[0] = correct[0] + 1
      else:
        incorrect[0] = incorrect[0] + 1
    print('Video [' + video['file_name'] + '] stop duration = [' + str(video['start_time'] - video['stop_time']) + '] long stop = [' + str(video['long_stop']) + '] predicted = [' + str(image_class) + ']')

if args.regression:
  print(f'Total videos [{str(len(video_test))}')
  print(f'Total error [{str(total_error_sec)}]')
  print(f'Mean error [{str(total_error_sec / len(video_test))}]')
  print(f'Max prediction [{str(max_duration)}]')
  print(f'Min prediction [{str(min_duration)}]')
else:
  print(f'Correct: short: {str(correct[0])}, long: {str(correct[1])}')
  print(f'Incorrect: short: {str(incorrect[0])}, long: {str(incorrect[1])}')
  accuracy = sum(correct) / (sum(correct) + sum(incorrect))
  print(f'Accuracy {accuracy}')