import random
import carla
import time
from DashcamMovementTracker import DashcamMovementTracker
import cv2

MAP = 'Town02'
ACTOR_CLASS = 'vehicle.citroen.c3'
ANY_VEHICLE_CLASS = 'vehicle.*'
CAMERA_CLASS = 'sensor.camera.rgb'
NUM_VEHICLES = 10
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

def get_blueprint(world, blueprint):
  bp_list = world.get_blueprint_library().filter(blueprint)
  bp_list = random.choice(bp_list)
  print(bp_list)
  return bp_list

try:
  client = carla.Client('localhost', 2000)
  client.set_timeout(20.0)
  world = client.load_world(MAP)
  spawn_points = world.get_map().get_spawn_points()
  waypoint_list = world.get_map().generate_waypoints(10.0)

  create_vehicles(world, NUM_VEHICLES)
  time.sleep(2)

  actor, camera = create_actor(world)
  
  #camera.listen(lambda image: image.save_to_disk('output/%06d.png' % image.frame))
  #spectator = world.get_spectator()
  #spectator.set_transform(actor.get_transform())



  #traffic_manager = client.get_trafficmanager(8000)
  #traffic_manager.set_global_distance_to_leading_vehicle(2.5)
  #settings = world.get_settings()
  #world.apply_settings(settings)

  #actor = spawn_driving_vehicle(client, world)
  #print(actor)
  time.sleep(5)

  #camera0 = carla.Camera('CameraRGB')
  #camera0.set_image_size(1280, 720)
  #camera0.set_position(0.30, 0, 1.30)
  #settings.add_sensor(camera0)

  #scene = client.load_settings(settings)

  #number_of_player_starts = len(scene.player_start_spots)
  #player_start = random.randint(0, max(0, number_of_player_starts-1))

  #client.start_episode(player_start)

  #for frame_index in range(0, frames_per_episode):
    #measurements, sensor_data = client.read_data()
    #print_measurements(measurements)

    #control = measurements.player_measurements.autopilot_control
    #control.steer += random.uniform(-0.1, 0.1)

    #client.send_control(control)
finally:
  client.stop_recorder()

print(captured_frames[0].raw_data)
print(len(captured_frames))

movement_tracker = DashcamMovementTracker()
movement_tracker.fps = CAM_FRAMES_PER_SECOND

for index in range(min(len(captured_frames), 10)):
  captured_frame = captured_frames[index]
  print(captured_frame.frame)
  captured_frame.save_to_disk('temp.png')
  reloaded_image = cv2.cvtColor(cv2.imread('temp.png'), cv2.COLOR_RGBA2RGB)
  movement_tracker.frames.append(reloaded_image)
  movement_tracker.frame_times.append(captured_frame.timestamp)

movement_tracker.write_video('test.mp4')