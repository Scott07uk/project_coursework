import json
import pathlib
from sys import platform as _platform
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm.notebook import trange, tqdm
import ffmpeg
import cv2
import psycopg2
import random

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

  #unused
  def get_min_stop_duration_ms(self):
    return self.__get_value_or_default('minStopDurationMS', 1000)
  
  #unused
  def get_meta_data_dir(self):
    return self.__get_value_or_default('metadataDir', '')

  #unused
  def get_file_limit_prefix(self):
    return self.__get_value_or_default('fileLimitPrefix', None)

  def get_db_user(self):
    return self.__get_value_or_default('dbUser', 'project')
  
  def get_db_pass(self):
    return self.__get_value_or_default('dbPass', 'password')

  def get_db_host(self):
    return self.__get_value_or_default('dbHost', 'oseidon.worldsofwar.co.uk')

  def get_db_port(self):
    return self.__get_value_or_default('dbPort', 5432)
  
  def get_db_name(self):
    return self.__get_value_or_default('dbName', 'dev')

  def get_db_url(self):
    return f'postgres://{self.get_db_user()}:{self.get_db_pass()}@{self.get_db_host()}:{self.get_db_port()}/{self.get_db_name()}'

  def get_psycopg2_conn(self):
    return f'host=\'{self.get_db_host()}\' dbname=\'{self.get_db_name()}\' user=\'{self.get_db_user()}\' password=\'{self.get_db_pass()}\''

  def get_temp_dir(self):
    return self.__get_value_or_default('tmpDir', '/home/scott/tmp')

  def get_windows_temp_dir(self):
    return self.__get_value_or_default('windowsTmpDir', None)

  def clean_temp(self):
    if self.__get_value_or_default('cleanTmpDir', True):
      self.__rm_tree(self.get_temp_dir(), True)

  def __rm_tree(self, pth, parent=False):
    pth = pathlib.Path(pth)
    for child in pth.glob('*'):
      if child.is_file():
        child.unlink()
      else:
        self.__rm_tree(child)
    if not parent:
      pth.rmdir()

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

class StopTime:
  '''Immutable class to record when the vehicle stopped'''
  def __init__(self, stop_time_ms, start_time_ms, video_end_time_ms):
    self.__stop_time_ms = stop_time_ms
    self.__start_time_ms = start_time_ms
    self.__video_end_time_ms = video_end_time_ms
    self.detected_start_time = None
    self.detected_stop_time = None

  def get_stop_duration_ms(self):
    if self.__start_time_ms is None:
      return self.__video_end_time_ms - self.__stop_time_ms
    return self.__start_time_ms - self.__stop_time_ms

  def get_stop_time_ms(self):
    return self.__stop_time_ms
  
  def get_start_time_ms(self):
    return self.__start_time_ms

  def get_detected_stop_time_diff(self):
    if self.detected_stop_time is None:
      return none
    return abs(self.detected_stop_time - self.__stop_time_ms)
  
  def get_detected_start_time_diff(self):
    if self.detected_start_time is None and self.__start_time_ms is None:
      return 0
    if self.detected_start_time is None:
      return abs(self.__start_time_ms - self.__video_end_time_ms)  
    if self.__start_time_ms is None:
      return abs(self.detected_start_time - self.__video_end_time_ms)  
    return abs(self.detected_start_time - self.__start_time_ms)

  def to_tuple(self):
    return (self.__stop_time_ms, self.__start_time_ms)
  


class BDDVideo:
  def __init__(self, file_name, absoloute_file_name, file_type, start_time, end_time):
    self.__file_name = file_name
    self.__absolute_file_name = absoloute_file_name
    self.__file_type = file_type
    self.__start_time = start_time
    self.__end_time = end_time
    self.__stop_times = {}
  
  def get_start_time(self):
    return self.__start_time

  def record_stop_time(self, stop_type, stop_time, start_time):
    raw_stop_time = None
    raw_start_time = None
    if not stop_time is None:
      raw_stop_time = stop_time - self.__start_time
    if not start_time is None:
      raw_start_time = start_time - self.__start_time

    self.record_raw_stop_time(stop_type, raw_stop_time, raw_start_time)

  def record_raw_stop_time(self, stop_type, stop_time, start_time):
    '''Record a stop time in true GPS time, this will convert it to video time'''
    if not stop_type in self.__stop_times.keys():
      self.__stop_times[stop_type] = []
    self.__stop_times[stop_type].append(StopTime(stop_time, start_time, self.__end_time))

  def clean_short_stops(self, min_stop_time, min_stop_duration):
    for stop_type in self.__stop_times.keys():
      new_stop_times = []
      for stop_time in self.__stop_times[stop_type]:
        #We remove any stop where the video has not had enough time to play
        if (stop_time.get_stop_time_ms() + self.__start_time) - self.__start_time >= min_stop_time:
          if stop_time.get_stop_duration_ms() >= min_stop_duration:
            new_stop_times.append(stop_time)
      self.__stop_times[stop_type] = new_stop_times

  def get_stop_times(self, type):
    if not type in self.__stop_times.keys():
      return []
    return self.__stop_times[type]

  def get_stop_times_as_tuples(self, type):
    return [x.to_tuple() for x in self.get_stop_times(type)]

  def has_any_stop_times(self):
    count = 0
    for stop_type in self.__stop_times.keys():
      count += len(self.__stop_times[stop_type])
    return count

  def has_stop_times(self, type):
    return len(self.__stop_times[type]) > 0
  
  def get_absoloute_file_name(self):
    return self.__absolute_file_name

  def get_file_name(self):
    return self.__file_name

  def get_start_time(self):
    return self.__start_time

  def to_metadata(self):
    output = {}
    output['fileName'] = self.__file_name
    output['fileType'] = self.__file_type
    output['videoWorldStartTime'] = self.__start_time
    output['videoWorldEndTime'] = self.__end_time

    stops = {}
    output['stops'] = stops

    for stop_type in self.__stop_times.keys():
      output_stops = []
      for stop in self.__stop_times[stop_type]:
        output_stops.append({'stop_time': stop.get_stop_time_ms(), 'start_time': stop.get_start_time_ms()})
      stops[stop_type] = output_stops

    return output

def json_file_to_bdd_video(config, data_type, path_to_file):
  '''Loads a JSON file from BDD and converts it into the BDDVideo'''
  with open(path_to_file) as info_file_content:
    try:
      info_file = json.load(info_file_content)
      video_file = path_to_file.name.replace('json', 'mov')
      print(video_file)
      vid_data = BDDVideo(video_file, config.get_absoloute_path_of_video(data_type, video_file), data_type, info_file['startTime'], info_file['endTime'])
      current_stop_time = None
      if 'gps' in info_file:
        for gps_loc in info_file['gps']:
          if (gps_loc['speed'] <= config.get_stop_gps_speed()):
            if current_stop_time is None:
              #Vehicle has stoped
              current_stop_time = gps_loc['timestamp']
          elif not current_stop_time is None:
            #Vehicle has started
            vid_data.record_stop_time('gps', current_stop_time, gps_loc['timestamp'])
            current_stop_time = None
            
        if not current_stop_time is None:
          #Vehicle stoped in the video and did not start again
          vid_data.record_stop_time('gps', current_stop_time, None)

      current_stop_time = None
      if 'locations' in info_file:
        for location in info_file['locations']:
          if (location['speed'] <= config.get_stop_gps_speed()):
            if current_stop_time is None:
              current_stop_time = location['timestamp']
          elif not current_stop_time is None:
            vid_data.record_stop_time('location', current_stop_time, location['timestamp'])
            current_stop_time = None
        if not current_stop_time is None:
          #Vehicle stoped in the video and did not start again
          vid_data.record_stop_time('location', current_stop_time, None)

      #print(f'Cleaning [{len(vid_data.get_stop_times())}] stops')
      vid_data.clean_short_stops(config.get_min_play_time_before_stop(), config.get_min_stop_duration_ms())

      return vid_data
    except json.JSONDecodeError:
      print(f'File {info_file_content} is not a valid json file, please check')
      return None

def video_stops_from_database(config: BDDConfig, PCT_VALID: float = 0.2, MIN_DURATION: int = None):
  CLASSIFIER_THRESH = 8000
  video_train = []
  video_valid = []
  video_all = []
  seen_file_names = []
  random.seed(42)
  db = psycopg2.connect(config.get_psycopg2_conn())
  cursor = db.cursor()
  sql = "SELECT file_name, file_type, stop_time_ms, start_time_ms, (start_time_ms - stop_time_ms) as duration FROM video_file INNER JOIN video_file_stop ON (id = video_file_id) WHERE stop_time_ms > 4000 and start_time_ms is not null AND state = 'DONE' AND stop_time_ms < start_time_ms"
  if MIN_DURATION is not None:
    sql = f'{sql} AND (start_time_ms - stop_time_ms) >= {MIN_DURATION}'
  cursor.execute(sql)
  row = cursor.fetchone()

  while row is not None:
    video = {
      'file_name': row[0],
      'file_type': row[1],
      'stop_time': float(row[2]),
      'start_time': float(row[3]),
      'duration': float(row[4]),
      'long_stop': row[4] >= CLASSIFIER_THRESH,
      'type': 'BDD'
    }

    if video['file_name'] not in seen_file_names:
      seen_file_names.append(video['file_name'])
      if random.random() < PCT_VALID:
        video_valid.append(video)
      else:
        video_train.append(video)
      video_all.append(video)

    row = cursor.fetchone()

  cursor.close()
  db.close()

  return video_train, video_valid