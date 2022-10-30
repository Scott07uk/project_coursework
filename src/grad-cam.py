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

IMAGENET_STATS = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
FRAME_SIZE = (int(720/2), int(1280/2))

class DensenetClassificationImageModel(pytorch_lightning.LightningModule):
  def __init__(self):
    super(DensenetClassificationImageModel, self).__init__()
    self.model = torchvision.models.densenet121()
    self.model.classifier = torch.nn.Linear(in_features=1024, out_features=2)

  def forward(self, x):
    return self.model(x)


transforms = torchvision.transforms.Compose(
  [
    torchvision.transforms.Resize(FRAME_SIZE),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(*IMAGENET_STATS)
  ]
)

class ActivationHolder():

  def __init__(self, heatmap_layer):
    self.gradient = None
    self.activation = None
    self.heatmap_layer = heatmap_layer
    heatmap_layer.register_forward_hook(self.hook)

  def set_gradient(self, grad):
    self.gradient = grad

  def hook(self, model, input, output):
    output.register_hook(self.set_gradient)
    self.activation = output.detach()

model = DensenetClassificationImageModel.load_from_checkpoint('models/single-frame-e127-a0.6426380276679993.ckpt')
model.eval()



def GradCAM(img, c, features_fn, classifier_fn):
  feats = features_fn(img)
  _, N, H, W = feats.size()
  print(feats.size())
  out = classifier_fn(feats)
  c_score = out[0, c]
  grads = torch.autograd.grad(c_score, feats)
  w = grads[0][0].mean(-1).mean(-1)
  sal = torch.matmul(w, feats.view(N, H*W))
  sal = sal.view(H, W).detach().numpy()
  sal = numpy.maximum(sal, 0)
  return sal

class Flatten(torch.nn.Module):
  """One layer module that flattens its input."""
  def __init__(self):
      super(Flatten, self).__init__()
  def forward(self, x):
      return x.view(x.size(0), -1)

image_file = '/mnt/ssd/scott/temp/bdd-still/1177c1ab-2ecc7a6e.mov-4200.0/19.jpeg'
image = Image.open(image_file)

input_tensor = transforms(image)
input_tensor = input_tensor.unsqueeze(0)
image_class = 1



#features_function = model.model.features
#classifier_function = torch.nn.Sequential(*([torch.nn.AvgPool2d(11, 10), Flatten()] + [model.model.classifier]))

#model_output = model(input_tensor)
#model_softmax = torch.nn.Softmax(dim=1)(model_output)
#print(model_output)
#print(model_softmax)
#pp, cc = torch.topk(model_softmax, 2)

#for i, (p, c) in enumerate(zip(pp[0], cc[0])):
#  print(p)
#  print(c)
#  sal = GradCAM(input_tensor, int(c), features_function, classifier_function)
#  image = cv2.imread(image_file)
#  sal = numpy.uint8(255 * sal)
#  print(sal)
#  sal = cv2.resize(sal, (image.shape[1], image.shape[0]))
#  sal = cv2.applyColorMap(sal, cv2.COLORMAP_JET)
#  superimposed_img = numpy.uint8(sal * 0.6 + image * 0.4)
#  cv2.imwrite(f'test{str(i)}.jpeg', superimposed_img)


from cam import GradCAM, ScoreCAM, SmoothGradCAMpp, reverse_normalize, visualize

img = reverse_normalize(input_tensor)
print('1')
target_layer = model.model.features.denseblock4.denselayer16.conv2
print('2')
wrapped_model = SmoothGradCAMpp(model.model, target_layer)
print('3')
cam, idx = wrapped_model(input_tensor)
print('4')
heatmap = visualize(img, cam)
print('5')
torchvision.utils.save_image(heatmap, 'scorecam.jpeg')
exit()
print('6')
wrapped_model = GradCAM(model.model, target_layer)
print('7')
cam, idx = wrapped_model(input_tensor)
print('8')
heatmap = visualize(img, cam)
print('9')
torchvision.utils.save_image(heatmap, 'gradcam.jpeg')
