#
# This file is to pre-process the data to work out if the vehicle is actually stationary
#

from BDD import BDDConfig, debug
import json
import cv2

CFG_FILE = 'cfg/laptop.json'

CONFIG = BDDConfig(CFG_FILE)
DEBUG = True

def validate_stationary_video(vid_data):
  if vid_data['stop_times'] is None or len(vid_data['stop_times']) == 0:
    return
  #debug(f'About to load {vid_data}')
  file_name = vid_data['absoloute_file_name']
  capture = cv2.VideoCapture(file_name)
  stop = vid_data['stop_times'][0]
  relative_stop_time = stop['stop_time'] - vid_data['start_time']
  print(f'Manually check {file_name} for stop {relative_stop_time / 1000} seconds into the video')
  #frame_number = 0
  #while (capture.isOpened()):
    #next_frame_exists, next_frame = capture.read()
    #if frame_exists:
      #print(f'Frame number {str(frame_number)} is {capture.get(cv2.CAP_PROP_POS_MSEC)}')
    #else:
      #break
    
    #frame_number += 1


files_loaded = 0
for info_file_path in CONFIG.get_info_dir_ls():
  with open(info_file_path) as info_file_content:
    files_loaded = files_loaded + 1
    #print(info_file_path.name)
    try:
      info_file = json.load(info_file_content)
      video_file = info_file['filename']
      vid_data = {}
      vid_data['start_time'] = info_file['startTime']
      min_stop_time = info_file['startTime'] + CONFIG.get_min_play_time_before_stop();
      vid_data['file_name'] = info_file_path.name.replace('json', 'mov')
      vid_data['absoloute_file_name'] = CONFIG.get_absoloute_path_of_video('train', vid_data['file_name'])
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
      
        stop_times = []
        for stop in vid_data['stop_times']:
          if stop['stop_time'] >= min_stop_time:
            stop_times.append(stop)
        vid_data['stop_times'] = stop_times

        validate_stationary_video(vid_data)
    except json.JSONDecodeError:
      print(f'File {info_file_content} is not a valid json file, please check')
     
  if files_loaded >= CONFIG.get_max_files_to_read():
    exit()
