#
# This file is to pre-process the data to work out if the vehicle is actually stationary
#

from BDD import BDDConfig, debug, run_function_in_parallel
import json
import cv2

CFG_FILE = 'cfg/laptop.json'

CONFIG = BDDConfig(CFG_FILE)
DEBUG = True

class StopTime:
  '''Immutable class to record when the vehicle stopped'''
  def __init__(self, stop_time_ms, start_time_ms):
    self.__stop_time_ms = stop_time_ms
    self.__start_time_ms = start_time_ms

  def get_stop_duration_ms(self):
    return self.__start_time - self.__stop_time

  def get_stop_time_ms(self):
    return self.__stop_time_ms
  
  def get_start_time_ms(self):
    return self.__start_time_ms


class BDDVideo:
  def __init__(self, file_name, absoloute_file_name, start_time):
    self.__file_name = file_name
    self.__absolute_file_name = absoloute_file_name
    self.__start_time = start_time
    self.__stop_times = []
  
  def get_start_time(self):
    return self.__start_time

  def record_stop_time(self, stop_time, start_time):
    self.__stop_times.append(StopTime(stop_time, start_time))

  def clean_short_stops(self, min_stop_time):
    new_stop_times = []
    for stop_time in self.__stop_times:
      #We remove any stop where the video has not had enough time to play
      if stop_time.get_stop_time_ms() - self.__start_time >= min_stop_time:
        new_stop_times.append(stop_time)
    self.__stop_times = new_stop_times

  def get_stop_times(self):
    return self.__stop_times

  def has_stop_times(self):
    return len(self.__stop_times) > 0
  
  def get_absoloute_file_name(self):
    return self.__absolute_file_name

  def get_file_name(self):
    return self.__file_name

  def get_start_time(self):
    return self.__start_time


def validate_stationary_video(vid_data):
  if not vid_data.has_stop_times():
    return
  #debug(f'About to load {vid_data}')
  capture = cv2.VideoCapture(vid_data.get_absoloute_file_name())
  stop = vid_data.get_stop_times()[0]
  relative_stop_time = stop.get_stop_time_ms() - vid_data.get_start_time()
  print(f'Manually check {vid_data.get_file_name()} for stop {relative_stop_time / 1000} seconds into the video')
  #frame_number = 0
  #while (capture.isOpened()):
    #next_frame_exists, next_frame = capture.read()
    #if frame_exists:
      #print(f'Frame number {str(frame_number)} is {capture.get(cv2.CAP_PROP_POS_MSEC)}')
    #else:
      #break
    
    #frame_number += 1

def load_and_validate_info_file(absoloute_path_to_info_file):
  with open(absoloute_path_to_info_file) as info_file_content:
    try:
      info_file = json.load(info_file_content)
      video_file = info_file_path.name.replace('json', 'mov')
      vid_data = BDDVideo(video_file, CONFIG.get_absoloute_path_of_video('train', video_file), info_file['startTime'])
      current_stop_time = None
      if 'gps' in info_file:
        for gps_loc in info_file['gps']:
          if (gps_loc['speed'] <= CONFIG.get_stop_gps_speed()):
            if current_stop_time is None:
              #Vehicle has stoped
              current_stop_time = gps_loc['timestamp']
          elif not current_stop_time is None:
            #Vehicle has started
            vid_data.record_stop_time(current_stop_time, gps_loc['timestamp'])
            
        if not current_stop_time is None:
          #Vehicle stoped in the video and did not start again
          vid_data.record_stop_time(current_stop_time, info_file['endTime'])
    
        vid_data.clean_short_stops(CONFIG.get_min_play_time_before_stop())

        validate_stationary_video(vid_data)
        if vid_data.has_stop_times():
          return vid_data
        return None
    except json.JSONDecodeError:
      print(f'File {info_file_content} is not a valid json file, please check')
      return None



info_files = []
for info_file_path in CONFIG.get_info_dir_ls():
  info_files.append(info_file_path)
  if len(info_files) >= CONFIG.get_max_files_to_read():
    break
print(f'Going to load {len(info_files)} info files')

checked_videos = []

if CONFIG.get_workers() == 1:
  # Lets not do any threading
  for info_file in info_files:
    result = load_and_validate_info_file(info_file)
    if not result is None:
      checked_videos.append(result)
else:
  # Lets do some multi threading
  output = run_function_in_parallel(load_and_validate_info_file, info_files, workers=CONFIG.get_workers())
  for result in output:
    if not result is None:
      checked_videos.append(result)

print(f'Found {len(checked_videos)} videos with stops')
