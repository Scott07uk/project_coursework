#
# This file is to pre-process the data to work out if the vehicle is actually stationary
#

from BDD import BDDConfig
import json

CFG_FILE = 'cfg/laptop.json'

CONFIG = BDDConfig(CFG_FILE)


for info_file_path in CONFIG.get_info_dir_ls():
  with open(info_file_path) as info_file_content:
    info_file = json.load(info_file_content)
    video_file = info_file['filename']
    stop_times = []
    current_stop = None
    for gps_loc in info_file['gps']:
      if (gps_loc['speed'] <= CONFIG.get_stop_gps_speed()):
        if current_stop is None:
          current_stop = {'stop_time': gps_loc['timestamp']}
          stop_times.append(current_stop)
      elif not current_stop is None:
        current_stop['start_time'] = gps_loc['timestamp']
        current_stop = None
    if not current_stop is None:
      current_stop['start_time'] = info_file['end_time']
    print(stop_times)

    exit()
