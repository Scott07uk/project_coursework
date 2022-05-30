import cv2
import os
import glob
import time
from DashcamMovementTracker import DashcamMovementTracker
from BDD import get_video_rotation

INPUT_DIR = '/home/scott/ownCloud/Video/Dashcam/New/'
OUTPUT_DIR = 'output/'
TIME_DIFF_THRESHOLD = 1.5

files = list(filter(os.path.isfile, glob.glob(INPUT_DIR + "*")))
files.sort(key=lambda x: os.path.getmtime(x))

if (len(files)) == 0:
  print('There are no files to join together')
  exit()

file_collectives = []
previous_file_time = None
current_collective = []
for file in files:
  file_ctime = os.path.getctime(file)
  if not previous_file_time is None:
    file_time_diff = file_ctime - previous_file_time
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
  output_filename = OUTPUT_DIR + os.path.basename(collective[0])
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
  exit()