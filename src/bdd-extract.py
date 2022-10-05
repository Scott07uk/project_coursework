from argparse import (
  ArgumentParser
)
from BDD import (
  BDDConfig,
  video_stops_from_database
)
from DashcamMovementTracker import DashcamMovementTracker
import pathlib
import cv2

parser = ArgumentParser()

parser.add_argument('--config', dest='config', action='store', help='The config file to use')
parser.add_argument('--perform-extract', dest='perform_extract', action='store_true', help='Perform the extract process from the original source videos')
parser.add_argument('--process-all', dest='process_all', action='store_true', help='Process all the videos rather than just the missing ones')

parser.set_defaults(config = 'cfg/kastria-local.json', perform_extract=False, process_all = False)
args = parser.parse_args()

CONFIG = BDDConfig(args.config)

video_train, video_test = video_stops_from_database(CONFIG)

video_all = video_train + video_test


if args.perform_extract:
  for video in video_all:
    video_file = CONFIG.get_absoloute_path_of_video(video['file_type'], video['file_name'])
    still_dir = CONFIG.get_temp_dir() + '/bdd-still/' + video['file_name'] + '-' + str(video['stop_time'])
    multi_still_dir = CONFIG.get_temp_dir() + '/bdd-multi-still/' + video['file_name'] + '-' + str(video['stop_time'])
    still_dir_path = pathlib.Path(still_dir)
    multi_still_dir = pathlib.Path(multi_still_dir)
    process = False
    if not still_dir_path.exists():
      still_dir_path.mkdir()
      process = True

    if not multi_still_dir.exists():
      multi_still_dir.mkdir()
      process = True

    for index in range(20):
      still_file_path = pathlib.Path(f'{still_dir}/{str(index)}.jpeg')
      multi_still_file_path = pathlib.Path(f'{multi_still_dir}/{str(index)}.jpeg')
      if (not still_file_path.exists()) or (not multi_still_file_path.exists()):
        process = True

    
    if (process or args.process_all):
      print(f'Processing video {video_file}')
      movement_tracker = DashcamMovementTracker()
      movement_tracker.get_video_frames_from_file(video_file)
      output = movement_tracker.get_training_data(video['stop_time'])

      stills = output['stills']
      multi_stills = output['multi-stills']
      for index in range(len(stills)):
        output_image_name = still_dir + '/' + str(index) + '.jpeg'
        cv2.imwrite(output_image_name, stills[index])

        output_image_name = multi_still_dir + '/' + str(index) + '.jpeg'
        cv2.imwrite(output_image_name, multi_stills[index])


