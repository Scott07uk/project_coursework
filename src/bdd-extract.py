from argparse import (
  ArgumentParser
)
from BDD import (
  BDDConfig,
  video_stops_from_database
)
from DashcamMovementTracker import DashcamMovementTracker
import cv2
import io
import matplotlib.pyplot
import numpy
import pandas
import pathlib
import PIL
import pytorch_lightning
import random
import seaborn
import torch.utils.data
import torchmetrics
import torchvision


IMAGENET_STATS = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


parser = ArgumentParser()
parser = pytorch_lightning.Trainer.add_argparse_args(parser)
parser.add_argument('--config', dest='config', action='store', help='The config file to use')
parser.add_argument('--perform-extract', dest='perform_extract', action='store_true', help='Perform the extract process from the original source videos')
parser.add_argument('--process-all', dest='process_all', action='store_true', help='Process all the videos rather than just the missing ones')
parser.add_argument('--dense-optical-flow', dest='dense_optical_flow', action='store_true', help='When extracting images, also extract the dense optical flow')
parser.add_argument('--sparse-optical-flow', dest='sparse_optical_flow', action='store_true', help='When extracting images, also extract the sparse optical flow')
parser.add_argument('--train-sparse-optical-flow', dest='train_sparse_optical_flow', action='store_true', help='Train a model with the sparse optical flow images')
parser.add_argument('--train-dense-optical-flow', dest='train_dense_optical_flow', action='store_true', help='Train a model with the dense optical flow images')


parser.set_defaults(config = 'cfg/kastria-local.json', perform_extract=False, process_all = False, dense_optical_flow = False, sparse_optical_flow = False, train_sparse_optical_flow = False, train_dense_optical_flow = False)
args = parser.parse_args()

CONFIG = BDDConfig(args.config)

video_train, video_test = video_stops_from_database(CONFIG)

video_all = video_train + video_test


if args.perform_extract:
  for video in video_all:
    video_file = CONFIG.get_absoloute_path_of_video(video['file_type'], video['file_name'])
    still_dir = CONFIG.get_temp_dir() + '/bdd-still/' + video['file_name'] + '-' + str(video['stop_time'])
    multi_still_dir = CONFIG.get_temp_dir() + '/bdd-multi-still/' + video['file_name'] + '-' + str(video['stop_time'])
    short_video_file = CONFIG.get_temp_dir() + '/bdd-video/' + video['file_name'] + '-' + str(video['stop_time']) + '.mp4'
    dense_optical_flow_still_dir = CONFIG.get_temp_dir() + '/bdd-dense-optical-flow/' + str(video['file_name']) + '-' + str(video['stop_time'])
    sparse_optical_flow_still_dir = CONFIG.get_temp_dir() + '/bdd-sparse-optical-flow/' + str(video['file_name']) + '-' + str(video['stop_time'])
    still_dir_path = pathlib.Path(still_dir)
    multi_still_dir_path = pathlib.Path(multi_still_dir)
    short_video_file_path = pathlib.Path(short_video_file)
    dense_optical_flow_still_dir_path = pathlib.Path(dense_optical_flow_still_dir)
    sparse_optical_flow_still_dir_path = pathlib.Path(sparse_optical_flow_still_dir)
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
    if args.sparse_optical_flow:
      if not sparse_optical_flow_still_dir_path.exists():
        sparse_optical_flow_still_dir_path.mkdir()
        process = True

    for index in range(20):
      still_file_path = pathlib.Path(f'{still_dir}/{str(index)}.jpeg')
      multi_still_file_path = pathlib.Path(f'{multi_still_dir}/{str(index)}.jpeg')
      sparse_optical_flow_file_path = pathlib.Path(f'{sparse_optical_flow_still_dir}/{str(index)}.jpeg')
      dense_optical_flow_file_path = pathlib.Path(f'{dense_optical_flow_still_dir}/{str(index)}.jpeg')
      if (not still_file_path.exists()) or (not multi_still_file_path.exists()) or (args.sparse_optical_flow and not sparse_optical_flow_file_path.exists()) or (args.dense_optical_flow and not dense_optical_flow_file_path.exists()):
        process = True

    
    if (process or args.process_all):
      print(f'Processing video {video_file}')
      movement_tracker = DashcamMovementTracker()
      movement_tracker.get_video_frames_from_file(video_file)
      output = movement_tracker.get_training_data(video['stop_time'], dense_optical_flow=args.dense_optical_flow, sparse_optical_flow=args.sparse_optical_flow)

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

      movement_tracker.write_video(short_video_file)


#From here on we look at training

class ImageModel(pytorch_lightning.LightningModule):
  def __init__(self, name = None, trainer = None):
    super(ImageModel, self).__init__()
    self.model = torchvision.models.densenet121(pretrained=True)
    print(self.model)
    self.model.classifier = torch.nn.Linear(in_features=1024, out_features=2)

    self.val_confusion = torchmetrics.ConfusionMatrix(num_classes=2)
    self.loss_weights = torch.FloatTensor([2.3, 4.15]).cuda()

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
    matplotlib.pyplot.figure()
    seaborn.set(font_scale=1.2)
    seaborn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d')
    buf = io.BytesIO()
    matplotlib.pyplot.savefig(buf, format='jpeg')
    buf.seek(0)
    im = PIL.Image.open(buf)
    im = torchvision.transforms.ToTensor()(im)
    tb.add_image("val_confusion_matrix", im, global_step=self.current_epoch)
    self.val_confusion.reset()
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


class ImageDataset(torch.utils.data.Dataset):
  def __init__(self, training, directory):
    self.directory = directory
    self.training = training
    self.frame_size = (int(720/2), int(1280/2))

    if training:
      self.data = video_train
      
      self.transforms = torchvision.transforms.Compose(
        [
          torchvision.transforms.ColorJitter(0.2, 0.2),
          torchvision.transforms.RandomHorizontalFlip(0.3),
          torchvision.transforms.Resize(self.frame_size),
          torchvision.transforms.ToTensor(),
          torchvision.transforms.Normalize(*IMAGENET_STATS)
        ]
      )
    else:
      self.data = video_test
      self.transforms = torchvision.transforms.Compose(
        [
          torchvision.transforms.Resize(self.frame_size),
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
    image_file = self.directory + '/' + video['file_name'] + '-' + str(video['stop_time']) + '/' + str(image_file_index) + '.jpeg'
    image = PIL.Image.open(image_file)
    duration_class = 0
    if video['long_stop']:
      duration_class = 1

    return self.transforms(image), duration_class

class ImageDataModule(pytorch_lightning.LightningDataModule):
  def __init__(self, directory):
    super().__init__()
    self.NUM_WORKERS = 5
    self.BATCH_SIZE = 12
    self.directory = directory

  def prepare_data(self):
    print('prepare_data')

  def train_dataloader(self):
    return torch.utils.data.DataLoader(ImageDataset(True, self.directory), batch_size=self.BATCH_SIZE, shuffle=True, num_workers=self.NUM_WORKERS)

  def val_dataloader(self):
    return torch.utils.data.DataLoader(ImageDataset(False, self.directory), batch_size=self.BATCH_SIZE, shuffle=False, num_workers=self.NUM_WORKERS)

if args.train_sparse_optical_flow:
  data_module = ImageDataModule(CONFIG.get_temp_dir() + '/bdd-sparse-optical-flow')
  trainer = pytorch_lightning.Trainer.from_argparse_args(args)
  model = ImageModel(name='sparse-optical-flow', trainer=trainer)
  model.cuda()
  trainer.fit(model, data_module)

if args.train_dense_optical_flow:
  data_module = ImageDataModule(CONFIG.get_temp_dir() + '/bdd-dense-optical-flow')
  trainer = pytorch_lightning.Trainer.from_argparse_args(args)
  model = ImageModel(name='dense-optical-flow', trainer=trainer)
  model.cuda()
  trainer.fit(model, data_module)