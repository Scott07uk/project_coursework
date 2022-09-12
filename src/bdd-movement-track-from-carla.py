from argparse import (
  ArgumentParser
)
from BDD import (BDDConfig)
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
parser.add_argument('--file-limit', dest='file-limit', action='store', help='Process this many files (default 10)')
parser.add_argument('--dry-run', dest='dry_run', action='store', help='Perform a dry run and output the videos with movement detection (default true)')
parser.add_argument('--config', dest='config', action='store', help='The config file to use')
parser.add_argument('--extract-frames', dest='extract_frames', action='store', help='Extract this many videos from BDD to add to carla')

parser.set_defaults(file_limit = 10, dry_run = True, config = 'cfg/kastria-local.json', extract_frames = 0)
args = parser.parse_args()

CONFIG = BDDConfig(args.config)

def run_inference(cv_input_image):
  pil_input_image = Image.fromarray(cv_input_image)
  input_tensor = transforms(pil_input_image)
  input_tensor = input_tensor.unsqueeze(0)
  output_tensor = classifier.forward(input_tensor)
  output_tensor = output_tensor[0]
  if output_tensor[0] > output_tensor[1]:
    return 0
  else:
    print('stationary')
  return 1

if args.extract_frames == 0:

  classifier = ImageModel.load_from_checkpoint(args.model_file)

  processed_file_count = 0
  path_to_videos = pathlib.Path(CONFIG.get_video_dir('train'))

  for video_file in path_to_videos.iterdir():
    movement_tracker = DashcamMovementTracker()
    movement_tracker.get_video_frames_from_file(str(video_file))
    frame_index = 0
    frame_stop_status = False
    movement_tracker.frame_stop_status = [False] * len(movement_tracker.frames)
    for second in range(4, int(max(movement_tracker.frame_times) / 1000)):
      while frame_index < len(movement_tracker.frames) and movement_tracker.frame_times[frame_index] < second:
        movement_tracker.frame_stop_status[frame_index] = frame_stop_status
        frame_index = frame_index + 1
      if frame_index >= len(movement_tracker.frames):
        break;
      blue = cv2.cvtColor(movement_tracker.frames[frame_index], cv2.COLOR_BGR2GRAY)
      green = cv2.cvtColor(movement_tracker.frames[frame_index - int(movement_tracker.fps * 2)], cv2.COLOR_BGR2GRAY)
      red = cv2.cvtColor(movement_tracker.frames[frame_index - int(movement_tracker.fps * 4)], cv2.COLOR_BGR2GRAY)
      output_image = numpy.dstack([red, green, blue]).astype(numpy.uint8)
      frame_stop_status = run_inference(output_image) == 0
    movement_tracker.write_video(f'test-{processed_file_count}.mp4', include_timings=True)
    processed_file_count = processed_file_count + 1
    if args.file_limit == -1 or processed_file_count >= args.file_limit:
      exit()

else:
  SQL = f"SELECT id, file_type, file_name FROM video_file vf INNER JOIN video_file_stop vfs ON (vf.id = vfs.video_file_id) WHERE stop_time_ms > 4000 and start_time_ms - stop_time_ms > 1000 ORDER BY video_file_id LIMIT {args.extract_frames}"

  with psycopg2.connect(CONFIG.get_psycopg2_conn()) as db:
    with db.cursor() as cursor:
      video_files = []
      cursor.execute(SQL)
      row = cursor.fetchone()
      while row is not None:
        video_file = {'file_id': row[0],
                      'file_type': row[1],
                      'file_name': row[2]}
        video_files.append(video_file)
        row = cursor.fetchone()
      
      for video_file in video_files:

        movement_tracker = DashcamMovementTracker()
        movement_tracker.get_video_frames_from_file(CONFIG.get_video_dir(video_file['file_type']) + '/' + video_file['file_name'])
        frame_index = 0
        for second in range(4, int(max(movement_tracker.frame_times) / 1000)):
          while frame_index < len(movement_tracker.frames) and movement_tracker.frame_times[frame_index] < second:
            frame_index = frame_index + 1
          if frame_index >= len(movement_tracker.frames):
            break;

          blue = cv2.cvtColor(movement_tracker.frames[frame_index], cv2.COLOR_BGR2GRAY)
          green = cv2.cvtColor(movement_tracker.frames[frame_index - int(movement_tracker.fps * 2)], cv2.COLOR_BGR2GRAY)
          red = cv2.cvtColor(movement_tracker.frames[frame_index - int(movement_tracker.fps * 4)], cv2.COLOR_BGR2GRAY)
          output_image = numpy.dstack([red, green, blue]).astype(numpy.uint8)
          
          dir_name = 'stopped'
          file_id = video_file['file_id']
          second_ms = second * 1000
          SQL = f"SELECT video_file_id FROM video_file_stop WHERE video_file_id = {file_id} AND stop_time_ms <= {second_ms} AND (start_time_ms >= {second_ms} OR start_time_ms IS NULL)"
          cursor.execute(SQL)
          row = cursor.fetchone()
          if row is None:
            dir_name = 'moving'

          output_file = CONFIG.get_temp_dir() + '/carla-movement/' + dir_name + '/bdd-' + video_file['file_name'] + '-' + str(second) + '.jpeg'
          cv2.imwrite(output_file, output_image)
