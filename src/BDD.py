import json
import pathlib

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

  def get_info_dir(self, type):
    return self.__config['infoDir'] + '/' + type
  
  def get_video_dir(self, type):
    return self.__config['videoDir'] + '/' + type

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


DEBUG=True
def debug(message):
  if DEBUG == True:
    print(message)
