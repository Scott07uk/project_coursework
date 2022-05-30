import cv2
import os
import glob
import time
from DashcamMovementTracker import DashcamMovementTracker
from BDD import get_video_rotation
import json
import shutil

INPUT_DIR = '/home/scott/ownCloud/Video/Dashcam/New/'
OUTPUT_DIR = 'output/'
TIME_DIFF_THRESHOLD = 75
CONF_FILE = 'cfg/video_builder.json'

conf = {'processed_videos': []}

if os.path.exists(CONF_FILE):
  with open(CONF_FILE) as conf_file_content:
    conf = json.load(conf_file_content)

files = list(filter(os.path.isfile, glob.glob(INPUT_DIR + "*")))
files.sort(key=lambda x: os.path.getmtime(x))

if (len(files)) == 0:
  print('There are no files to join together')
  exit()

new_files = []
for file in files:
  processed_videos = conf['processed_videos']
  if not os.path.basename(file) in processed_videos:
    new_files.append(file)

files = new_files

file_collectives = []
previous_file_time = None
current_collective = []
for file in files:
  file_ctime = os.path.getmtime(file)
  #print(f'{file}: {file_ctime}')
  if not previous_file_time is None:
    file_time_diff = file_ctime - previous_file_time
    #print(file_time_diff)
    if file_time_diff <= TIME_DIFF_THRESHOLD:
      current_collective.append(file)
    else:
      file_collectives.append(current_collective)
      current_collective = [file]
  else:
    current_collective.append(file)
  previous_file_time = file_ctime

file_collectives.append(current_collective)

for collective_index in range(len(file_collectives)):
  collective = file_collectives[collective_index]
  print(f'Going to join videos {collective}')

  output_filename = OUTPUT_DIR + os.path.basename(collective[0])
  if len(collective) == 1:
    #we can just copy the video file
    shutil.copy(src=collective[0], dst=output_filename)
  else:
    video_rotation = get_video_rotation(collective[0])
    capture = cv2.VideoCapture(collective[0])
    fps = capture.get(cv2.CAP_PROP_FPS)
    next_frame_exists, next_frame = capture.read()
    if not video_rotation is None:
      #some videos have screwed up meta data
      if next_frame.shape[0] > next_frame.shape[1]:
        next_frame = cv2.rotate(next_frame, video_rotation)
    height, width, channels = next_frame.shape
    vid_size = (width, height)
  
    print(f'Writing video {output_filename} from {len(collective)} parts')
    out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'MP4V'), fps, vid_size)
    capture.release()
    for file_index in range(len(collective)):
      video_rotation = get_video_rotation(collective[file_index])
      capture = cv2.VideoCapture(collective[file_index])
      counter = 0
      while (capture.isOpened()):
        next_frame_exists, next_frame = capture.read()
        if next_frame_exists:
          if not video_rotation is None:
            #some videos have screwed up meta data
            if next_frame.shape[0] > next_frame.shape[1]:
              next_frame = cv2.rotate(next_frame, video_rotation)
          out.write(next_frame)
          counter += 1
        else:
          break
      capture.release()
    
    out.release()

  for file_name in collective:
    conf['processed_videos'].append(os.path.basename(file_name))

  json_str = json.dumps(conf, indent = 2)
  with open(CONF_FILE, "w") as outfile:
    outfile.write(json_str)