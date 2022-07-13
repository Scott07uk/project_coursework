import random
import carla
import time
from DashcamMovementTracker import DashcamMovementTracker
import cv2
import numpy

MAP = 'Town02'
ACTOR_CLASS = 'vehicle.citroen.c3'
ANY_VEHICLE_CLASS = 'vehicle.*'
ANY_PERSON_CLASS = 'walker.pedestrian.*'
CAMERA_CLASS = 'sensor.camera.rgb'
NUM_VEHICLES = 10
NUM_PEOPLE = 10
CAM_FRAMES_PER_SECOND = 10
CAM_IMAGE_SIZE_X = 1280
CAM_IMAGE_SIZE_Y = 720
CAM_FIELD_OF_VIEW = 140
CAM_RELATIVE_LOCATION = carla.Location(x=0.7, y=0.1, z=1.35)

captured_frames = []


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
  camera = world.spawn_actor(camera_bp, carla.Transform(CAM_RELATIVE_LOCATION), attach_to=actor)
  camera.listen(lambda image: captured_frames.append(image))
  
  spectator = world.get_spectator()
  spectator.set_transform(carla.Transform(spawn_point.location + CAM_RELATIVE_LOCATION, spawn_point.rotation))

  actor.set_autopilot(True)
  return actor, camera

def create_vehicles(world, number = 20):
  spawn_points = world.get_map().get_spawn_points()
  vehicles = []
  for index in range(number):
    spawn_point = random.choice(spawn_points)
    vehicle_bp = get_blueprint(world, ANY_VEHICLE_CLASS)
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if vehicle is not None:
      vehicle.set_autopilot(True)
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

try:
  client = carla.Client('kastria.worldsofwar.co.uk', 2000)
  client.set_timeout(20.0)
  world = client.load_world(MAP)
  spawn_points = world.get_map().get_spawn_points()
  waypoint_list = world.get_map().generate_waypoints(10.0)

  create_vehicles(world, NUM_VEHICLES)
  create_people(world, NUM_PEOPLE)
  time.sleep(2)

  actor, camera = create_actor(world)
  
  time.sleep(30)

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

#print(dir(captured_frames[0].raw_data.tobytes()))
#print(captured_frames[0].raw_data.tobytes())
#data = numpy.frombuffer(captured_frames[0].raw_data.tobytes(), dtype=numpy.uint8)
#data = cv2.imdecode(data, cv2.IMREAD_COLOR)
#print(data)

movement_tracker = DashcamMovementTracker()
movement_tracker.fps = CAM_FRAMES_PER_SECOND

for index in range(len(captured_frames)):
  captured_frame = captured_frames[index]
  print(captured_frame.frame)
  reloaded_image = to_rgb_array(captured_frame)
  movement_tracker.frames.append(reloaded_image)
  movement_tracker.frame_times.append(captured_frame.timestamp)

movement_tracker.write_video('test.mp4')