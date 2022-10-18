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
parser.add_argument('--dense-optical-flow', dest='dense_optical_flow', action='store_true', help='When extracting images, also extract the dense optical flow')
parser.add_argument('--sparse-optical-flow', dest='sparse_optical_flow', action='store_true', help='When extracting images, also extract the sparse optical flo


parser.set_defaults(config = 'cfg/kastria-local.json', perform_extract=False, process_all = False, dense_optical_flow = False, sparse_optical_flow = False)
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

        if args.dense_optical_flow:
          optical_flow_stills = output['dense-optical-flow-stills']
          output_image_name = dense_optical_flow_still_dir + '/' + str(index) + '.jpeg'
          cv2.imwrite(output_image_name, optical_flow_stills[index])
          
        if args.sparse_optical_flow:
          optical_flow_stills = output['sparse-optical-flow-stills']
          output_image_name = sparse_optical_flow_still_dir + '/' + str(index) + '.jpeg'
          cv2.imwrite(output_image_name, optical_flow_stills[index])

      movement_tracker.write_video(short_video_file)


