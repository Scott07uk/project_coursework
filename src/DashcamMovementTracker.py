import cv2
import numpy
from BDD import get_video_rotation
import time
import pathlib
from os.path import exists

class DashcamMovementTracker:
  def __init__(self, track_region_width=150, shift_count_for_changes = 7, pixel_movement_thresh = 10, track_loss_threshold=0.92):
    self.__track_region_width = track_region_width
    self.__shift_count_for_changes = shift_count_for_changes
    self.__pixel_movement_thresh = pixel_movement_thresh
    self.__track_loss_threshold = track_loss_threshold
    self.fps = None
    self.frames = []
    self.frame_times = []
    self.frame_stop_status = []

  def get_stops_from_frames(self, frames, frame_times, debug=False):
    video_frame_shape = frames[0].shape
    #x, y, width, height
    orig_bbox_left = (0, 0, self.__track_region_width, video_frame_shape[0])
    bbox_left_range = [[0 - (video_frame_shape[1] * (1 - self.__track_loss_threshold)), 0 + (video_frame_shape[1] * (1 - self.__track_loss_threshold))], [0 - (video_frame_shape[0] * (1 - self.__track_loss_threshold)), 0 + (video_frame_shape[0] * (1 - self.__track_loss_threshold))], [self.__track_region_width * self.__track_loss_threshold, self.__track_region_width * (2 - self.__track_loss_threshold)], [video_frame_shape[0] * self.__track_loss_threshold, video_frame_shape[0] * (2 - self.__track_loss_threshold)]]
    orig_bbox_right = (video_frame_shape[1] - self.__track_region_width, 0, self.__track_region_width, video_frame_shape[0])
    bbox_right_range = [[orig_bbox_right[0] - (video_frame_shape[1] * (1 - self.__track_loss_threshold)), orig_bbox_right[0] + (video_frame_shape[1] * (1 - self.__track_loss_threshold))], [0 - (video_frame_shape[0] * (1 - self.__track_loss_threshold)), 0 + (video_frame_shape[0] * (1 - self.__track_loss_threshold))], [self.__track_region_width * self.__track_loss_threshold, self.__track_region_width * (2 - self.__track_loss_threshold)], [video_frame_shape[0] * self.__track_loss_threshold, video_frame_shape[0] * (2 - self.__track_loss_threshold)]]
    bbox_left = orig_bbox_left
    bbox_right = orig_bbox_right

    if debug:
      print(f'Input Shape {video_frame_shape}')
      print(f'Left bbox {bbox_left}')
      print(f'Right bbox {bbox_right}')

    tracker_left = cv2.TrackerCSRT_create()
    tracker_right = cv2.TrackerCSRT_create()
    ok_left = tracker_left.init(frames[0], bbox_left)
    ok_right = tracker_right.init(frames[0], bbox_right)

    pixel_shifts = []
    stop_time = None
    stops = []
    self.frame_stop_status = []

    for index in range(1, len(frames), 5):
      frame = frames[index]

      prev_bbox_left = bbox_left
      prev_bbox_right = bbox_right
      ok_left, bbox_left = tracker_left.update(frame)
      for check_index in range(4):
        if bbox_left[check_index] < bbox_left_range[check_index][0] or bbox_left[check_index] > bbox_left_range[check_index][1]:
          if debug:
            print(f'Lost too much height in left movement tracker, resetting to {orig_bbox_left}')
          bbox_left = orig_bbox_left
          tracker_left.init(frame, orig_bbox_left)
          break

      ok_right, bbox_right = tracker_right.update(frame)
      for check_index in range(4):
        if bbox_right[check_index] < bbox_right_range[check_index][0] or bbox_right[check_index] > bbox_right_range[check_index][1]:
          if debug:
            print(f'Lost too much height in right movement tracker, resetting to {orig_bbox_right}')
          bbox_right = orig_bbox_right
          tracker_right.init(frame, orig_bbox_right)
          break

      shift_left_squared = abs((prev_bbox_left[0] - bbox_left[0]) + (prev_bbox_left[1] - bbox_left[1]) + (prev_bbox_left[2] - bbox_left[2]) + (prev_bbox_left[3] - bbox_left[3]))
      shift_right_squared = abs((prev_bbox_right[0] - bbox_right[0]) + (prev_bbox_right[1] - bbox_right[1]) + (prev_bbox_right[2] - bbox_right[2]) + (prev_bbox_right[3] - bbox_right[3]))
      pixel_shifts.append((shift_left_squared, shift_right_squared))

      if debug:
        p1 = (int(bbox_left[0]), int(bbox_left[1]))
        p2 = (int(bbox_left[0] + bbox_left[2]), int(bbox_left[1] + bbox_left[3]))
        frame = frame.copy()
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)

        p1 = (int(bbox_right[0]), int(bbox_right[1]))
        p2 = (int(bbox_right[0] + bbox_right[2]), int(bbox_right[1] + bbox_right[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        print(f'Pixel shifts for frame at {frame_times[index]} = [{shift_left_squared + shift_right_squared}]')

        cv2.imshow("Tracking", frame)
        cv2.waitKey(1)

      shift_count = len(pixel_shifts)
      if shift_count > self.__shift_count_for_changes:
        if stop_time is None:
          should_stop = True
          for check in range(shift_count - self.__shift_count_for_changes - 1, shift_count -1):
            if pixel_shifts[check][0] + pixel_shifts[check][1] > self.__pixel_movement_thresh:
              should_stop = False
          if should_stop:
            #The vehicle has stopped
            stop_time = frame_times[index - ((self.__shift_count_for_changes + 1) * 5)]
        else:
          should_start = True
          for check in range(shift_count - self.__shift_count_for_changes - 1, shift_count -1):
            if pixel_shifts[check][0] < self.__pixel_movement_thresh and pixel_shifts[check][1] < self.__pixel_movement_thresh:
              should_start = False
          if should_start:
            #The vehcile has started moving
            stops.append((stop_time, frame_times[index - ((self.__shift_count_for_changes + 1) * 5)]))
            stop_time = None

      for tmp in range(5):
        self.frame_stop_status.append(not stop_time is None)

    if not stop_time is None:
      #The vehicle stoped, but did not start again before the video ended
      stops.append((stop_time, None))

    while len(self.frame_stop_status) < len(self.frames):
      #If we have a few frame left over at the end, add the last status on to complete the set
      self.frame_stop_status.append(self.frame_stop_status[len(self.frame_stop_status) - 1])

    self.stops = stops

    return stops

  def get_video_frames_from_file(self, file_name, debug=False):
    if not exists(file_name):
      print(f'File {file_name} does not exist, going to abort')
      return None
    video_rotation = get_video_rotation(file_name, debug=debug)
    capture = cv2.VideoCapture(file_name)
    if debug:
      print(f'Identified video rotation: {video_rotation} (90 clockwise = {cv2.ROTATE_90_CLOCKWISE}, 180 = {cv2.ROTATE_180}, 90 anti-clockwise = {cv2.ROTATE_90_COUNTERCLOCKWISE})')
    frame_times = []
    frames = []
    self.fps = capture.get(cv2.CAP_PROP_FPS)
    while (capture.isOpened()):
      next_frame_exists, next_frame = capture.read()
      if next_frame_exists:
        if not video_rotation is None:
          #some videos have screwed up meta data
          if next_frame.shape[0] > next_frame.shape[1]:
            next_frame = cv2.rotate(next_frame, video_rotation)
        frame_times.append(capture.get(cv2.CAP_PROP_POS_MSEC))
        frames.append(next_frame)
      else:
        break
    capture.release()

    self.frame_times = frame_times
    self.frames = frames
    return (frame_times, frames)

  def get_stops_from_file(self, file_name, debug=False):
    frames_and_times = self.get_video_frames_from_file(file_name, debug=debug)
    if frames_and_times is None:
      return None
    frame_times, frames = frames_and_times
    return self.get_stops_from_frames(frames, frame_times, debug=debug)

  def change_frame_rate(self, new_fps):
    time_between_frames = 1000 / new_fps
    new_frames = [self.frames[0]]
    new_frame_times = [self.frame_times[0]]

    for index in range(len(self.frames)):
      if new_frame_times[-1] + time_between_frames <= self.frame_times[index]:
        new_frames.append(self.frames[index])
        new_frame_times.append(self.frame_times[index])

    self.fps = new_fps
    self.frames = new_frames
    self.frame_times = new_frame_times

  def to_channel_time(self, red_time = 4000, green_time = 2000):
    single_channel_frames = []
    new_frames = []
    red_prev_frames = None
    green_prev_frames = None
    for index in range(len(self.frames)):
      if red_prev_frames is None:
        if self.frame_times[index] - red_time >= self.frame_times[0]:
          red_prev_frames = index
      if green_prev_frames is None:
        if self.frame_times[index] - green_time >= self.frame_times[0]:
          green_prev_frames = index

      if green_prev_frames is not None and red_prev_frames is not None:
        break

    for index in range(len(self.frames)):
      base_image = cv2.cvtColor(self.frames[index], cv2.COLOR_BGR2GRAY)
      single_channel_frames.append(base_image.copy())

      red_index = index - red_prev_frames
      green_index = index - green_prev_frames

      red = single_channel_frames[0]
      green = single_channel_frames[0]
      blue = base_image
      if red_index >= 0:
        red = single_channel_frames[red_index]
      if green_index >= 0:
        green = single_channel_frames[green_index]

      new_image = numpy.dstack([red, green, base_image]).astype(numpy.uint8)
      new_frames.append(new_image)
    self.frames = new_frames

  def cut(self, start_time = 0, end_time = 1000):
    new_frames = []
    new_times = []
    for index in range(len(self.frames)):
      if self.frame_times[index] >= start_time:
        if self.frame_times[index] <= end_time:
          new_frames.append(self.frames[index])
          new_times.append(self.frame_times[index] - start_time)
        else:
          break
    self.frames = new_frames
    self.frame_times = new_times
      

  def write_video(self, file_name, include_timings = False):
    height, width, layers = self.frames[0].shape
    vid_size = (width, height)
    out = cv2.VideoWriter(file_name ,cv2.VideoWriter_fourcc(*'DIVX'), self.fps, vid_size)
    moving = False
    current_stop_index = 0
    for frame_index in range(len(self.frames)):
      frame = self.frames[frame_index]
      if include_timings:
        if self.frame_stop_status[frame_index]:
          cv2.putText(frame, "STOPPED", (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,255,50), 2);
        else:
          cv2.putText(frame, "MOVING", (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,50,50), 2);
      out.write(frame)
    out.release()


DEBUG_STOP_TIMES = False
DEBUG_FRAME_RATE_CHANGE = False
DEBUG_CHANNEL_TIME = False
DIRECTORY = '/mnt/usb/bdd/bdd100k/videos/train/'

if DEBUG_STOP_TIMES:
  
  
  FILES_TO_TEST = []

  path = pathlib.Path(DIRECTORY)

  counter = 0
  for file in path.iterdir():
    FILES_TO_TEST.append(file.name)
    counter += 1
    if counter > 200:
      break

  for file in FILES_TO_TEST:
    dashcam_movement_tracker = DashcamMovementTracker()
    absoloute_file = f'{DIRECTORY}{file}'
    print(f'Going to load file {absoloute_file}')
    stops = dashcam_movement_tracker.get_stops_from_file(absoloute_file, debug=False)
    for stop in stops:
      print(f'  Found stop between {stop[0]} and {stop[1]}')
    dashcam_movement_tracker.write_video('/home/scott/test/' + file, include_timings = True)


if DEBUG_CHANNEL_TIME:
  FILES_TO_TEST = []

  path = pathlib.Path(DIRECTORY)

  counter = 0
  for file in path.iterdir():
    FILES_TO_TEST.append(file.name)
    counter += 1
    if counter > 10:
      break

  for file in FILES_TO_TEST:
    dashcam_movement_tracker = DashcamMovementTracker()
    absoloute_file = f'{DIRECTORY}{file}'
    print(f'Going to load file {absoloute_file}')
    dashcam_movement_tracker.get_video_frames_from_file(absoloute_file)
    dashcam_movement_tracker.to_channel_time()
    dashcam_movement_tracker.write_video('/home/scott/test/' + file)


if DEBUG_FRAME_RATE_CHANGE:
  FILES_TO_TEST = []

  path = pathlib.Path(DIRECTORY)

  counter = 0
  for file in path.iterdir():
    FILES_TO_TEST.append(file.name)
    counter += 1
    if counter > 1:
      break

  for file in FILES_TO_TEST:
    for fps in range(2, 14, 2):
      dashcam_movement_tracker = DashcamMovementTracker()
      absoloute_file = f'{DIRECTORY}{file}'
      print(f'Going to load file {absoloute_file}')
      dashcam_movement_tracker.get_video_frames_from_file(absoloute_file)
      dashcam_movement_tracker.change_frame_rate(fps)
      dashcam_movement_tracker.write_video('/home/scott/test/' + str(fps) + '-' + file)


cv2.destroyAllWindows()