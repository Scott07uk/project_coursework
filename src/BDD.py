import json
import pathlib

class BDDConfig:
  def __init__(self, config_file):
    json_file_handle = open(config_file)
    self.__config = json.load(json_file_handle)
    json_file_handle.close()

  def get_info_dir(self, type):
    return self.__config['infoDir'] + '/' + type

  def get_info_dir_ls(self, type = 'train'):
    path = pathlib.Path(self.get_info_dir(type))
    return path.iterdir()

  def get_stop_gps_speed(self):
    return self.__config['stopGPSSpeed']
  
  def get_max_files_to_read(self):
    return self.__config['maxReadFiles']
