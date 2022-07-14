import random
import carla
import time
from DashcamMovementTracker import DashcamMovementTracker
import cv2
import numpy
import re

MAP = 'Town02'
ACTOR_CLASS = 'vehicle.ford.crown'
ANY_VEHICLE_CLASS = 'vehicle.*'
ANY_PERSON_CLASS = 'walker.pedestrian.*'
CAMERA_CLASS = 'sensor.camera.rgb'
NUM_VEHICLES = 20
NUM_PEOPLE = 10
CAM_FRAMES_PER_SECOND = 20
CAM_IMAGE_SIZE_X = 1280
CAM_IMAGE_SIZE_Y = 720
CAM_FIELD_OF_VIEW = 140
CAM_RELATIVE_LOCATION = carla.Location(x=2.2, y=0.0, z=1.1)
OUTPUT_FRAMES_PER_SECOND = 15
MOVEMENT_THRESH = 0.01
WEATHER_BASE = 'ClearNoon'

captured_frames = []
captured_moving = []

def record_video_frame(actor, image):
  captured_frames.append(image)
  velocity = actor.get_velocity()
  captured_moving.append(abs(velocity.x) < MOVEMENT_THRESH and abs(velocity.y) < MOVEMENT_THRESH and abs(velocity.z) < MOVEMENT_THRESH)

def create_actor(world):
  spawn_points = world.get_map().get_spawn_points()
  while True:
    spawn_point = random.choice(spawn_points)
    if (spawn_point.rotation.yaw < 0.1) and (spawn_point.rotation.yaw > -0.1):
      break
  actor_bp = get_blueprint(world, ACTOR_CLASS)
  camera_bp = get_blueprint(world, CAMERA_CLASS)
  camera_bp.set_attribute('sensor_tick', str(1 / CAM_FRAMES_PER_SECOND))
  camera_bp.set_attribute('image_size_x', str(CAM_IMAGE_SIZE_X))
  camera_bp.set_attribute('image_size_y', str(CAM_IMAGE_SIZE_Y))
  camera_bp.set_attribute('fov', str(CAM_FIELD_OF_VIEW))

  actor = world.spawn_actor(actor_bp, spawn_point)
  light_mask = carla.VehicleLightState.Position
  light_mask |= carla.VehicleLightState.LowBeam
  light_mask |= carla.VehicleLightState.HighBeam
  
  camera = world.spawn_actor(camera_bp, carla.Transform(CAM_RELATIVE_LOCATION), attach_to=actor)
  camera.listen(lambda image: record_video_frame(actor, image))
  
  spectator = world.get_spectator()
  spectator.set_transform(carla.Transform(spawn_point.location + CAM_RELATIVE_LOCATION, spawn_point.rotation))

  actor.set_autopilot(True)
  actor.set_light_state(carla.VehicleLightState(light_mask))
  return actor, camera

def create_vehicles(world, number = 20):
  spawn_points = world.get_map().get_spawn_points()
  vehicles = []

  light_mask = carla.VehicleLightState.NONE
  light_mask |= carla.VehicleLightState.Position
  light_mask |= carla.VehicleLightState.Brake
  light_mask |= carla.VehicleLightState.LowBeam

  for index in range(number):
    spawn_point = random.choice(spawn_points)
    vehicle_bp = get_blueprint(world, ANY_VEHICLE_CLASS)
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if vehicle is not None:
      vehicle.set_autopilot(True)
      vehicle.set_light_state(carla.VehicleLightState(light_mask))
      vehicles.append(vehicle)
  return vehicles

def create_people(world, number=10):
  people = []
  for index in range(number):
    spawn_point = None
    while spawn_point is None:
      spawn_point = world.get_random_location_from_navigation()
    
    person_bp = get_blueprint(world, ANY_PERSON_CLASS)
    if person_bp.has_attribute('is_invincible'):
      person_bp.set_attribute('is_invincible', 'false')
    person = world.try_spawn_actor(person_bp, carla.Transform(spawn_point))
    if person is not None:
      people.append(person)

  return people

def get_blueprint(world, blueprint):
  bp_list = world.get_blueprint_library().filter(blueprint)
  bp_list = random.choice(bp_list)
  print(bp_list)
  return bp_list

def set_weather(world):
  weather = getattr(carla.WeatherParameters, WEATHER_BASE)
  #weather.fog_density = 100.0
  #weather.fog_distance = 10.0
  #weather.precipitation = 100.0
  #weather.precipitation_deposits = 100.
  #weather.wind_intensity = 100.0
  world.set_weather(weather)

try:
  client = carla.Client('localhost', 2000)
  client.set_timeout(20.0)
  world = client.load_world(MAP)
  rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
  def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
  presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
  print(presets)
  set_weather(world)

  create_vehicles(world, NUM_VEHICLES)
  create_people(world, NUM_PEOPLE)
  time.sleep(2)

  actor, camera = create_actor(world)
  
  time.sleep(10)
  camera.destroy()

finally:
  client.stop_recorder()


def to_bgra_array(image):
  array = numpy.frombuffer(image.raw_data, dtype=numpy.dtype("uint8"))
  array = numpy.reshape(array, (image.height, image.width, 4))
  return array

def to_rgb_array(image):
  array = to_bgra_array(image)
  # Convert BGRA to RGB.
  array = array[:, :, :3]
  array = array[:, :, ::-1]
  return array


movement_tracker = DashcamMovementTracker()
movement_tracker.fps = OUTPUT_FRAMES_PER_SECOND
video_time_sec = 0.0
captured_frame_count = len(captured_frames)
current_frame_index = 0
current_movement_index = 0
while True:
  this_frame_diff = abs(video_time_sec - captured_frames[current_frame_index].timestamp)
  next_frame_diff = abs(video_time_sec - captured_frames[current_frame_index + 1].timestamp)

  if this_frame_diff <= next_frame_diff:
    captured_frame = captured_frames[current_frame_index]
    reloaded_image = to_rgb_array(captured_frame)
    reloaded_image = cv2.cvtColor(reloaded_image, cv2.COLOR_RGB2BGR)
    movement_tracker.frames.append(reloaded_image)
    movement_tracker.frame_times.append(captured_frame.timestamp * 1000)  
    movement_tracker.frame_stop_status.append(captured_moving[current_frame_index])

  current_frame_index += 1
  video_time_sec += (1 / OUTPUT_FRAMES_PER_SECOND)

  if current_frame_index + 1 >= captured_frame_count:
    break

movement_tracker.write_video('test.mp4', include_timings=True)