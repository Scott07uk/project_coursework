#
# This file is to pre-process the data to work out if the vehicle is actually stationary
#

from BDD import BDDConfig
import json
import cv2

CFG_FILE = 'cfg/laptop.json'

CONFIG = BDDConfig(CFG_FILE)

files_loaded = 0
for info_file_path in CONFIG.get_info_dir_ls():
  with open(info_file_path) as info_file_content:
    files_loaded = files_loaded + 1
    print(info_file_path)
    info_file = json.load(info_file_content)
    video_file = info_file['filename']
    vid_data = {}
    vid_data['start_time'] = info_file['startTime']
    vid_data['file_name'] = info_file['filename']
    vid_data['stop_times'] = []
    current_stop = None
    if 'gps' in info_file:
      for gps_loc in info_file['gps']:
        if (gps_loc['speed'] <= CONFIG.get_stop_gps_speed()):
          if current_stop is None:
            current_stop = {'stop_time': gps_loc['timestamp']}
            vid_data['stop_times'].append(current_stop)
        elif not current_stop is None:
          current_stop['start_time'] = gps_loc['timestamp']
          current_stop = None
      if not current_stop is None:
        current_stop['start_time'] = info_file['endTime']
      print(vid_data)
     
  if files_loaded >= CONFIG.get_max_files_to_read():
    exit()
