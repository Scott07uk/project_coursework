import cv2
import numpy

SHIFT_COUNT_CHANGE = 6

VIDEOS = ['363771c2-1b968052.mov', '0fbfa814-80526065.mov', '1a5d0011-88de8a10.mov', '1e6ce86f-a11187bc.mov', '2e71f7e9-e144bc00.mov', '2e0166cc-27e85a3f.mov']

for vid_file in VIDEOS:

  video = cv2.VideoCapture("/mnt/usb/bdd/bdd100k/videos/train/" + vid_file)

  # Exit if video not opened.
  if not video.isOpened():
    print("Could not open video")
    sys.exit()

  frames = []
  frame_times = []
  while True:
    ok, frame = video.read()
    if not ok:
      break
    else:
      frames.append(frame)
      frame_times.append(video.get(cv2.CAP_PROP_POS_MSEC))

  video.release()

  # Define an initial bounding box
  bbox_left = (0, 0, 150, 720)
  bbox_right = (1280-150, 0, 150, 720)

  # Initialize tracker with first frame and bounding box
  tracker_left = cv2.TrackerCSRT_create()
  tracker_right = cv2.TrackerCSRT_create()
  ok_left = tracker_left.init(frames[0], bbox_left)
  ok_right = tracker_right.init(frames[0], bbox_right)
  shifts = []
  stop_time = None
  for index in range(1, len(frames), 5):
    # Read a new frame
    frame = frames[index]

    # Update tracker
    prev_bbox_left = bbox_left
    prev_bbox_right = bbox_right
    ok_left, bbox_left = tracker_left.update(frame)
    ok_right, bbox_eight = tracker_right.update(frame)
    shift_left = ((prev_bbox_left[0] - bbox_left[0]) + (prev_bbox_left[1] - bbox_left[1]) + (prev_bbox_left[2] - bbox_left[2]) + (prev_bbox_left[3] - bbox_left[3]))**2
    shift_right = ((prev_bbox_right[0] - bbox_right[0]) + (prev_bbox_right[1] - bbox_right[1]) + (prev_bbox_right[2] - bbox_right[2]) + (prev_bbox_right[3] - bbox_right[3]))**2
    shifts.append(shift_left + shift_right)
    #print(f'Time = {frame_times[index]} shifts = {shift_left + shift_right}')

    shift_count = len(shifts)
    if shift_count > SHIFT_COUNT_CHANGE:
      if stop_time is None:
        should_stop = True
        for check in range(shift_count - SHIFT_COUNT_CHANGE - 1, shift_count -1):
          if shifts[check] > 10:
            should_stop = False
        if should_stop:
          stop_time = frame_times[index - SHIFT_COUNT_CHANGE]
      else:
        should_start = True
        for check in range(shift_count - SHIFT_COUNT_CHANGE - 1, shift_count -1):
          if shifts[check] < 10:
            should_start = False
        if should_start:
          print(f'{vid_file}: stop between {stop_time} and {frame_times[index - SHIFT_COUNT_CHANGE]}')
          stop_time = None

  if not stop_time is None:
    print(f'{vid_file}: stop between {stop_time} and end of video')
