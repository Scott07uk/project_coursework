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
class ImageModel(pytorch_lightning.LightningModule):
  def __init__(self):
    super(ImageModel, self).__init__()
    self.model = torchvision.models.densenet121(pretrained=True)
    self.model.classifier = torch.nn.Linear(in_features=1024, out_features=2)

  def forward(self, x):
    out = self.model(x)
    return out

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

parser.set_defaults(config = 'cfg/kastria-local.json')
args = parser.parse_args()

CONFIG = BDDConfig(args.config)

classifier = ImageModel.load_from_checkpoint(args.model_file)

def run_inference(image_file_name):
  pil_input_image = Image.open(image_file_name)
  input_tensor = transforms(pil_input_image)
  input_tensor = input_tensor.unsqueeze(0)
  output_tensor = classifier.forward(input_tensor)
  output_tensor = output_tensor[0]
  if output_tensor[0] > output_tensor[1]:
    return 0
  return 1

video_train, video_test = video_stops_from_database(CONFIG)

correct = [0, 0]
incorrect = [0, 0]

for video in video_test:
  image_file_name = CONFIG.get_temp_dir() + '/bdd-multi-still/' + video['file_name'] + '-' + str(video['stop_time']) + '.jpeg'
  image_class = run_inference(image_file_name)
  
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

print(f'Correct: short: {str(correct[0])}, long: {str(correct[1])}')
print(f'Incorrect: short: {str(incorrect[0])}, long: {str(incorrect[1])}')
accuracy = sum(correct) / (sum(correct) + sum(incorrect))
print(f'Accuracy {accuracy}')