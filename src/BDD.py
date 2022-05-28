import json
import pathlib
from sys import platform as _platform
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm.notebook import trange, tqdm
import ffmpeg
import cv2

class BDDConfig:
  def __init__(self, config_file):
    json_file_handle = open(config_file)
    self.__config = json.load(json_file_handle)
    json_file_handle.close()

  def __get_value_or_default(self, value, default):
    if value in self.__config:
      return self.__config[value]
    else:
      return default

  def get_types_to_load(self):
    return self.__get_value_or_default('typesToLoad', ['train'])

  def get_info_dir(self, type):
    info_dirs = self.__config['infoDirs']
    info_dir = info_dirs[type]
    return info_dir + '/' + type
  
  def get_video_dir(self, type):
    video_dirs = self.__config['videoDirs']
    video_dir = video_dirs[type]
    return video_dir + '/' + type

  def get_info_dir_ls(self, type = 'train'):
    path = pathlib.Path(self.get_info_dir(type))
    return path.iterdir()

  def get_absoloute_path_of_video(self, type, file):
    return self.get_video_dir(type) + '/' + file

  def get_stop_gps_speed(self):
    return self.__config['stopGPSSpeed']
  
  def get_max_files_to_read(self):
    return self.__get_value_or_default('maxReadFiles', 10)

  def get_min_play_time_before_stop(self):
    '''Gets the minimum amount of time that a video must play before a stop is allowed'''
    return self.__get_value_or_default('minPlayTimeBeforeStop', 1000)
  
  def get_workers(self):
    return self.__get_value_or_default('workersToUse', 1)

  def get_min_stop_duration_ms(self):
    return self.__get_value_or_default('minStopDurationMS', 1000)


DEBUG=True
def debug(message):
  if DEBUG == True:
    print(message)

def run_function_in_parallel(func, params, workers=4):
  pool = ProcessPoolExecutor if _platform.startswith('linux') else ThreadPoolExecutor
  with pool(max_workers=workers) as ex:
    futures = [ex.submit(func,i) for i in params]
    results = [r for r in as_completed(futures)]  # results in random order
  res2ix = {v:k for k,v in enumerate(results)}
  out = [results[res2ix[f]].result() for f in futures]
  return out

def get_video_rotation(path_video_file, debug=False):
  # this returns meta-data of the video file in form of a dictionary
  try:
    meta_dict = ffmpeg.probe(path_video_file)
    tags = meta_dict['streams'][0]['tags']
    if not 'rotate' in tags.keys():
      return None
    meta_rotation = tags['rotate']
    if debug:
      print(f'Identified meta rotation of {meta_rotation}')
    if int(meta_rotation) == 90:
      return cv2.ROTATE_90_CLOCKWISE
    elif int(meta_rotation) == 180:
      return cv2.ROTATE_180
    elif int(meta_rotation) == 270:
      return cv2.ROTATE_90_COUNTERCLOCKWISE

    return None
  except ffmpeg.Error as e:
    print(e.stderr)
    exit()
