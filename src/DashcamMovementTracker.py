import cv2
import numpy
from BDD import get_video_rotation

class DashcamMovementTracker:
  def __init__(self, track_region_width=150, shift_count_for_changes = 6, pixel_movement_thresh = 10):
    self.__track_region_width = track_region_width
    self.__shift_count_for_changes = shift_count_for_changes
    self.__pixel_movement_thresh = pixel_movement_thresh

  def get_stops_from_frames(self, frames, frame_times):
    video_frame_shape = frames[0].shape
    #x, y, width, height
    bbox_left = (0, 0, self.__track_region_width, video_frame_shape[0])
    bbox_right = (video_frame_shape[1] - self.__track_region_width, 0, self.__track_region_width, video_frame_shape[0])

    tracker_left = cv2.TrackerCSRT_create()
    tracker_right = cv2.TrackerCSRT_create()
    ok_left = tracker_left.init(frames[0], bbox_left)
    ok_right = tracker_right.init(frames[0], bbox_right)

    pixel_shifts = []
    stop_time = None
    stops = []

    for index in range(1, len(frames), 5):
      frame = frames[index]

      prev_bbox_left = bbox_left
      prev_bbox_right = bbox_right
      ok_left, bbox_left = tracker_left.update(frame)
      ok_right, bbox_eight = tracker_right.update(frame)
      shift_left_squared = ((prev_bbox_left[0] - bbox_left[0]) + (prev_bbox_left[1] - bbox_left[1]) + (prev_bbox_left[2] - bbox_left[2]) + (prev_bbox_left[3] - bbox_left[3]))**2
      shift_right_squared = ((prev_bbox_right[0] - bbox_right[0]) + (prev_bbox_right[1] - bbox_right[1]) + (prev_bbox_right[2] - bbox_right[2]) + (prev_bbox_right[3] - bbox_right[3]))**2
      pixel_shifts.append(shift_left_squared + shift_right_squared)

      shift_count = len(pixel_shifts)
      if shift_count > self.__shift_count_for_changes:
        if stop_time is None:
          should_stop = True
          for check in range(shift_count - self.__shift_count_for_changes - 1, shift_count -1):
            if pixel_shifts[check] > self.__pixel_movement_thresh:
              should_stop = False
          if should_stop:
            #The vehicle has stopped
            stop_time = frame_times[index - self.__shift_count_for_changes]
        else:
          should_start = True
          for check in range(shift_count - self.__shift_count_for_changes - 1, shift_count -1):
            if pixel_shifts[check] < self.__pixel_movement_thresh:
              should_start = False
          if should_start:
            #The vehcile has started moving
            stops.append((stop_time, frame_times[index - self.__shift_count_for_changes]))
            stop_time = None

    if not stop_time is None:
      #The vehicle stoped, but did not start again before the video ended
      stops.append((stop_time, None))

    return stops

  def get_stops_from_file(self, file_name):
    video_rotation = get_video_rotation(file_name)
    capture = cv2.VideoCapture(file_name)

    frame_times = []
    frames = []

    while (capture.isOpened()):
      next_frame_exists, next_frame = capture.read()
      if next_frame_exists:
        if not video_rotation is None:
          next_frame = cv2.rotate(next_frame, video_rotation)
        frame_times.append(capture.get(cv2.CAP_PROP_POS_MSEC))
        frames.append(next_frame)
      else:
        break
    capture.release()

    return self.get_stops_from_frames(frames, frame_times)


FILES_TO_TEST = ['363771c2-1b968052.mov', '0fbfa814-80526065.mov', '1a5d0011-88de8a10.mov', '1e6ce86f-a11187bc.mov', '2e71f7e9-e144bc00.mov', '2e0166cc-27e85a3f.mov']
DIRECTORY = '/mnt/usb/bdd/bdd100k/videos/train/'

dashcam_movement_tracker = DashcamMovementTracker()

for file in FILES_TO_TEST:
  absoloute_file = f'{DIRECTORY}{file}'
  print(f'Going to load file {absoloute_file}')
  stops = dashcam_movement_tracker.get_stops_from_file(absoloute_file)
  for stop in stops:
    print(f'  Found stop between {stop[0]} and {stop[1]}')