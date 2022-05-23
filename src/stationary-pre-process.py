#
# This file is to pre-process the data to work out if the vehicle is actually stationary
#

from BDD import BDDConfig, debug, run_function_in_parallel, get_video_rotation
from os.path import exists
import json
import cv2
import numpy
import pathlib

CFG_FILE = '../cfg/laptop.json'

CONFIG = BDDConfig(CFG_FILE)
DEBUG = True

class StopTime:
  '''Immutable class to record when the vehicle stopped'''
  def __init__(self, stop_time_ms, start_time_ms):
    self.__stop_time_ms = stop_time_ms
    self.__start_time_ms = start_time_ms

  def get_stop_duration_ms(self):
    return self.__start_time_ms - self.__stop_time_ms

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

  def clean_short_stops(self, min_stop_time, min_stop_duration):
    new_stop_times = []
    for stop_time in self.__stop_times:
      #We remove any stop where the video has not had enough time to play
      if stop_time.get_stop_time_ms() - self.__start_time >= min_stop_time:
        if stop_time.get_stop_duration_ms() >= min_stop_duration:
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


def get_closest_frame_to_time(times, frames, time):
  current_distance = 10000000000
  for index in range(len(times)):
    difference = times[index] - time
    if difference < 0:
      difference = 0 - difference
    if current_distance > difference:
      current_distance = difference
    elif current_distance < difference: 
      print(f'Returning frame at {index - 1}')
      return index
  return len(times) -1

def validate_stationary_video(vid_data):
  if not vid_data.has_stop_times():
    return None
  #debug(f'About to load {vid_data}')
  video_filename = vid_data.get_absoloute_file_name()
  if not exists(video_filename):
    debug(f'File {video_filename} does not exist, going to skip')
    return None
  video_rotation = get_video_rotation(video_filename)
  capture = cv2.VideoCapture(video_filename)

  frame_times = []
  frames = []

  while (capture.isOpened()):
    next_frame_exists, next_frame = capture.read()
    if not video_rotation is None:
      next_frame = cv2.rotate(next_frame, video_rotation)
    if next_frame_exists:
      frame_times.append(capture.get(cv2.CAP_PROP_POS_MSEC) + vid_data.get_start_time())
      frames.append(next_frame)
    else:
      break
  capture.release()

  if len(frames) == 0:
    # Not possible to read the video file, might be corrupt
    return None
  
  for vid_segment_index in range(len(vid_data.get_stop_times())):
    process_video_segment(vid_data, vid_segment_index, frame_times, frames)
  return vid_data

def process_video_segment(vid_data, vid_segment_index, frame_times, frames):
  USE_DENSE = False
  if USE_DENSE:
    return process_video_segment_dense_optical_flow(vid_data, vid_segment_index, frame_times, frames)
  return process_video_segment_sparse_optical_flow(vid_data, vid_segment_index, frame_times, frames)

def process_video_segment_dense_optical_flow(vid_data, vid_segment_index, frame_times, frames):
  print(f'Processing segment {vid_segment_index} in {vid_data.get_absoloute_file_name()}')
  stop = vid_data.get_stop_times()[vid_segment_index]
  print(f'Stop times {stop.get_stop_time_ms()} start time {stop.get_start_time_ms()}, duration {stop.get_stop_duration_ms()}')
  stop_index = get_closest_frame_to_time(frame_times, frames, stop.get_stop_time_ms())
  start_index = get_closest_frame_to_time(frame_times, frames, stop.get_start_time_ms())
  print(f'stop index = {stop_index}, start index = {start_index}')
  prev_frame = cv2.cvtColor(frames[stop_index], cv2.COLOR_BGR2GRAY)
  hsv = numpy.zeros_like(frames[stop_index])
  hsv[...,1] = 255
  output_frames = []
  print(f'Going to process video {vid_data.get_absoloute_file_name()} between {stop_index} and {start_index}')
  for index in range(stop_index + 1, start_index+1, 3):
    #print(f'processing frame {index}')
    next_frame = cv2.cvtColor(frames[index], cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0) 
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])   
    hsv[...,0] = ang*180/numpy.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    prev_frame = next_frame
    output_frame = numpy.hstack((frames[index], rgb))
    output_frames.append(output_frame)

  write_video_to_filesystem(vid_data, vid_segment_index, output_frames)
  print('done')

fix_frame = lambda fr: fr[:,:,::-1]

def process_video_segment_sparse_optical_flow(vid_data, vid_segment_index, frame_times, frames):
  print(f'Processing segment {vid_segment_index} in {vid_data.get_absoloute_file_name()}')
  stop = vid_data.get_stop_times()[vid_segment_index]
  print(f'Stop times {stop.get_stop_time_ms()} start time {stop.get_start_time_ms()}, duration {stop.get_stop_duration_ms()}')
  stop_index = get_closest_frame_to_time(frame_times, frames, stop.get_stop_time_ms())
  start_index = get_closest_frame_to_time(frame_times, frames, stop.get_start_time_ms())
  print(f'stop index = {stop_index}, start index = {start_index}')
  prev_frame = cv2.cvtColor(frames[stop_index], cv2.COLOR_BGR2GRAY)
  p0 = cv2.goodFeaturesToTrack(prev_frame, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
  output_frames = []
  mask = numpy.zeros_like(frames[stop_index])
  colours = numpy.random.randint(0,255,(100,3))
  print(f'Going to process video {vid_data.get_absoloute_file_name()} between {stop_index} and {start_index}')
  for index in range(stop_index + 1, start_index+1, 1):
    #print(f'processing frame {index}')
    next_frame = cv2.cvtColor(frames[index], cv2.COLOR_BGR2GRAY)
    p1,st,_ = cv2.calcOpticalFlowPyrLK(prev_frame, next_frame, p0, None, winSize=(15,15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    good_new, good_old = p1[st==1], p0[st==1]
    
    for i,(new,old) in enumerate(zip(good_new,good_old)):
      a,b = new.ravel()
      c,d = old.ravel()
      mask = cv2.line(mask, (a,b),(c,d), colours[i].tolist(), 2)
      frame = cv2.circle(frames[index],(a,b),5,colours[i].tolist(),-1)
    frame = cv2.add(frame,mask)
    output_frames.append(frame)

    prev_frame = next_frame.copy()
    p0 = good_new.reshape(-1,1,2)

  write_video_to_filesystem(vid_data, vid_segment_index, output_frames)
  print('done')

def write_video_to_filesystem(vid_data, vid_segment_index, frames):
  height, width, layers = frames[0].shape
  vid_size = (width, height)
  out = cv2.VideoWriter('/home/scott/test/' + vid_data.get_file_name().replace('.mov', '-' + str(vid_segment_index) + '.mov') ,cv2.VideoWriter_fourcc(*'DIVX'), 15, vid_size)
  for frame in frames:
    out.write(frame)
  out.release()

def load_and_validate_info_file_wrapper(data_type_and_absoloute_path_to_info_file):
  data_type, absoloute_path_to_info_file = data_type_and_absoloute_path_to_info_file
  load_and_validate_info_file(data_type, absoloute_path_to_info_file)

def load_and_validate_info_file(data_type, absoloute_path_to_info_file):
  with open(absoloute_path_to_info_file) as info_file_content:
    try:
      info_file = json.load(info_file_content)
      video_file = absoloute_path_to_info_file.name.replace('json', 'mov')
      vid_data = BDDVideo(video_file, CONFIG.get_absoloute_path_of_video(data_type, video_file), info_file['startTime'])
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
            current_stop_time = None
            
        if not current_stop_time is None:
          #Vehicle stoped in the video and did not start again
          vid_data.record_stop_time(current_stop_time, info_file['endTime'])
    
        vid_data.clean_short_stops(CONFIG.get_min_play_time_before_stop(), CONFIG.get_min_stop_duration_ms())

        if vid_data.has_stop_times():
          print(f'Validating [{len(vid_data.get_stop_times())}] stops')
          vid_data = validate_stationary_video(vid_data)
          return vid_data
        return None
    except json.JSONDecodeError:
      print(f'File {info_file_content} is not a valid json file, please check')
      return None

#print(load_and_validate_info_file('train', pathlib.Path('/mnt/usb/bdd/bdd100k/info/100k/train/0178ff32-d94d3509.json')))
#exit()


info_files = []
for data_type in CONFIG.get_types_to_load():
  for info_file_path in CONFIG.get_info_dir_ls():
    info_files.append((data_type, info_file_path))
    if len(info_files) >= CONFIG.get_max_files_to_read():
      break


print(f'Going to load {len(info_files)} info files')

checked_videos = []

if CONFIG.get_workers() == 1:
  # Lets not do any threading
  for data_type, info_file in info_files:
    result = load_and_validate_info_file(data_type, info_file)
    if not result is None:
      checked_videos.append(result)
else:
  # Lets do some multi threading
  output = run_function_in_parallel(load_and_validate_info_file_wrapper, info_files, workers=CONFIG.get_workers())
  for result in output:
    if not result is None:
      checked_videos.append(result)

print(f'Found {len(checked_videos)} videos with stops')
