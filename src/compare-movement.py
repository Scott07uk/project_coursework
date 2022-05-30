from BDD import BDDConfig, json_file_to_bdd_video, run_function_in_parallel
from DashcamMovementTracker import DashcamMovementTracker
import cv2
from os.path import exists

CFG_FILE = 'cfg/laptop.json'

CONFIG = BDDConfig(CFG_FILE)

try_index = 2

def build_and_write_stop_video(frames, frame_times, fps, output_file_name, stops, font_colours=None):
  default_colour = (50,255,50)
  starting_pos = (100, 0)
  y_increment = 50
  height, width, layers = frames[0].shape
  vid_size = (width, height)
  out = cv2.VideoWriter(output_file_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, vid_size)
  for frame_index in range(len(frames)):
    frame = frames[frame_index]
    frame_time = frame_times[frame_index]
    text_pos = starting_pos
    
    for stop_type in stops.keys():
      stop = stops[stop_type]
      text_pos = (text_pos[0], text_pos[1] + y_increment)
      text = f'{stop_type}: MOVING'
      for time in stop:
        #print(f'Frame time {frame_time}, {time[0]}, {time[1]}')
        if frame_time >= time[0] and (time[1] is None or frame_time <= time[1]):
          text = f'{stop_type}: STOPPED'
          break;
      
      colour = default_colour
      if (not font_colours is None) and (stop_type in font_colours):
        colour = font_colours[stop_type]
      
      cv2.putText(frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2);

    out.write(frame)
  out.release()

def process_video(json_file_name):
  bdd_video = json_file_to_bdd_video(CONFIG, 'train', json_file_name)

  if exists(bdd_video.get_absoloute_file_name()):
    movement_tracker = DashcamMovementTracker()
    detected_stops = movement_tracker.get_stops_from_file(bdd_video.get_absoloute_file_name())
    stop_times = {
      'GPS': bdd_video.get_stop_times_as_tuples('gps'),
      'MOTION': detected_stops
    }

    build_and_write_stop_video(movement_tracker.frames, movement_tracker.frame_times, movement_tracker.fps, '/home/scott/test/' + bdd_video.get_file_name(), stop_times)
  else:
    print(f'File {bdd_video.get_absoloute_file_name()} does not exist')


json_files = []
for json_file_name in CONFIG.get_info_dir_ls():

  json_files.append(json_file_name)

  if len(json_files) >= CONFIG.get_min_stop_duration_ms():
    break

if CONFIG.get_workers() == 1:
  # Lets not do any threading
  for json_file_name in json_files:
    result = process_video(json_file_name)
else:
  # Lets do some multi threading
  output = run_function_in_parallel(process_video, json_files, workers=CONFIG.get_workers())
