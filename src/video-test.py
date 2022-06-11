import os
import pytorch_lightning
import pytorchvideo.data
import torch.utils.data

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip
)

import pytorchvideo.models.resnet

import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
from pytorchvideo.data.encoded_video import EncodedVideo

valid_transform = Compose(
  [
    ApplyTransformToKey(
      key="video",
      transform=Compose(
        [
          UniformTemporalSubsample(8),
          Lambda(lambda x: x / 255.0),
          Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
        ]
      ),
    ),
  ]
)

class KineticsDataModule(pytorch_lightning.LightningDataModule):

  # Dataset configuration
  _DATA_PATH = '/mnt/ssd/scott/videos'
  _CLIP_DURATION = 2  # Duration of sampled clip for each video
  _BATCH_SIZE = 4
  _NUM_WORKERS = 5  # Number of parallel processes fetching data

  def train_dataloader(self):
    """
    Create the Kinetics train partition from the list of video labels
    in {self._DATA_PATH}/train.csv. Add transform that subsamples and
    normalizes the video before applying the scale, crop and flip augmentations.
    """
    train_transform = Compose(
      [
      ApplyTransformToKey(
        key="video",
        transform=Compose(
          [
            UniformTemporalSubsample(8),
            Lambda(lambda x: x / 255.0),
            Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
            RandomShortSideScale(min_size=256, max_size=320),
            RandomCrop(244),
            RandomHorizontalFlip(p=0.5),
          ]
        ),
      ),
      ]
    )
    print('bob')
    train_dataset = pytorchvideo.data.Kinetics(
      data_path=os.path.join(self._DATA_PATH, "train"),
      clip_sampler=pytorchvideo.data.make_clip_sampler("random", self._CLIP_DURATION),
      transform=train_transform
    )
    print(train_dataset)
    return torch.utils.data.DataLoader(
      train_dataset,
      batch_size=self._BATCH_SIZE,
      num_workers=self._NUM_WORKERS,
    )

  def val_dataloader(self):
    """
    Create the Kinetics validation partition from the list of video labels
    in {self._DATA_PATH}/val
    """
    
    print('bob2')
    data_path = os.path.join(self._DATA_PATH, "valid")
    print(data_path)
    valid_dataset = pytorchvideo.data.Kinetics(
      data_path=data_path,
      clip_sampler=pytorchvideo.data.make_clip_sampler("random", self._CLIP_DURATION),
      transform=valid_transform
    )
    print(valid_dataset)
    return torch.utils.data.DataLoader(
      valid_dataset,
      batch_size=self._BATCH_SIZE,
      num_workers=self._NUM_WORKERS,
    )

def make_kinetics_resnet():
  return pytorchvideo.models.resnet.create_resnet(
      input_channel=3, # RGB input from Kinetics
      model_depth=50, # For the tutorial let's just use a 50 layer network
      model_num_class=2, # Kinetics has 400 classes so we need out final head to align
      norm=nn.BatchNorm3d,
      activation=nn.ReLU,
  )


class VideoClassificationLightningModule(pytorch_lightning.LightningModule):
  def __init__(self):
    super().__init__()
    self.model = make_kinetics_resnet()

  def forward(self, x):
    return self.model(x)

  def training_step(self, batch, batch_idx):
    # The model expects a video tensor of shape (B, C, T, H, W), which is the
    # format provided by the dataset
    y_hat = self.model(batch["video"])

    # Compute cross entropy loss, loss.backwards will be called behind the scenes
    # by PyTorchLightning after being returned from this method.
    loss = F.cross_entropy(y_hat, batch["label"])

    # Log the train loss to Tensorboard
    self.log("train_loss", loss.item())

    return loss

  def validation_step(self, batch, batch_idx):
    y_hat = self.model(batch["video"])
    loss = F.cross_entropy(y_hat, batch["label"])
    self.log("val_loss", loss)
    return loss

  def configure_optimizers(self):
    """
    Setup the Adam optimizer. Note, that this function also can return a lr scheduler, which is
    usually useful for training video models.
    """
    return torch.optim.Adam(self.parameters(), lr=1e-1)

def train(args):
  classification_module = VideoClassificationLightningModule()
  data_module = KineticsDataModule()
  trainer = pytorch_lightning.Trainer.from_argparse_args(args)
  classification_module.cuda()
  trainer.fit(classification_module, data_module)


def main(args):
  #train(args)

  classifier = VideoClassificationLightningModule.load_from_checkpoint('/home/scott/Documents/fh/project_coursework/lightning_logs/version_1/checkpoints/epoch=16-step=935.ckpt')
  test_classifier(classifier)

def test_classifier(classifier):
  video = EncodedVideo.from_path('/mnt/ssd/scott/videos/valid/day/cb540cff-6928e2e0.mp4')

  video_data = video.get_clip(start_sec=1, end_sec=3)

  video_data = valid_transform(video_data)

  inputs = video_data["video"]
  inputs = [i.to('cpu')[None, ...] for i in inputs]
  preds = classifier(inputs)
  print(preds)



if __name__ == "__main__":
  parser = ArgumentParser()
  parser = pytorch_lightning.Trainer.add_argparse_args(parser)
  args = parser.parse_args()

  main(args)
