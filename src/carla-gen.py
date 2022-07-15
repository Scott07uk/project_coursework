import random
import carla
import time
from DashcamMovementTracker import DashcamMovementTracker
import cv2
import numpy
import re
from BDD import BDDConfig
import psycopg2

CONFIG = BDDConfig('cfg/kastria-local.json')
ACTOR_CLASS = 'vehicle.ford.crown'
ANY_VEHICLE_CLASS = 'vehicle.*'
ANY_PERSON_CLASS = 'walker.pedestrian.*'
CAMERA_CLASS = 'sensor.camera.rgb'
CAM_FRAMES_PER_SECOND = 25
CAM_IMAGE_SIZE_X = 1280
CAM_IMAGE_SIZE_Y = 720
CAM_FIELD_OF_VIEW = 140
CAM_RELATIVE_LOCATION = carla.Location(x=2.2, y=0.0, z=1.1)
SPECTATOR_RELATIVE_LOCATION = carla.Location(x=0, y=0.0, z=80)
OUTPUT_FRAMES_PER_SECOND = 15
MOVEMENT_THRESH = 0.01

captured_frames = []
captured_moving = []

def record_video_frame(actor, image):
  captured_frames.append(image)
  velocity = actor.get_velocity()
  captured_moving.append(abs(velocity.x) < MOVEMENT_THRESH and abs(velocity.y) < MOVEMENT_THRESH and abs(velocity.z) < MOVEMENT_THRESH)

def create_actor(world, light_mask):
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

  actor = None
  while actor is None:
    actor = world.try_spawn_actor(actor_bp, spawn_point)
    time.sleep(2)
  
  camera = world.spawn_actor(camera_bp, carla.Transform(CAM_RELATIVE_LOCATION), attach_to=actor)
  
  spectator = world.get_spectator()
  #spectator.set_transform(carla.Transform(spawn_point.location + CAM_RELATIVE_LOCATION, spawn_point.rotation))
  spectator.set_transform(carla.Transform(spawn_point.location + SPECTATOR_RELATIVE_LOCATION, carla.Rotation(pitch=-90)))

  actor.set_autopilot(True)
  actor.set_light_state(carla.VehicleLightState(light_mask))

  camera.listen(lambda image: record_video_frame(actor, image))
  return actor, camera

def create_vehicles(world, number, light_mask):
  spawn_points = world.get_map().get_spawn_points()
  vehicles = []

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
  #print(bp_list)
  return bp_list

def set_weather(settings, world):
  weather = getattr(carla.WeatherParameters, settings['weather_base'])
  weather.fog_density = settings['fog_density']
  weather.fog_distance = settings['fog_distance']
  weather.precipitation = settings['precipitation']
  weather.precipitation_deposits = settings['precipitation_deposits']
  weather.wind_intensity = settings['wind_intensity']
  world.set_weather(weather)

  light_mask = carla.VehicleLightState.NONE
  if 'Night' in settings['weather_base']:
    light_mask |= carla.VehicleLightState.Position
    light_mask |= carla.VehicleLightState.LowBeam
    light_mask |= carla.VehicleLightState.HighBeam
    if settings['fog_distance'] < 40:
      light_mask |= carla.VehicleLightState.Fog
  elif settings['fog_distance'] < 40:
    light_mask |= carla.VehicleLightState.Position
    light_mask |= carla.VehicleLightState.LowBeam
    light_mask |= carla.VehicleLightState.Fog

  return light_mask



def run_simulation(settings):
  try:
    captured_frames = []
    captured_moving = []
    client = carla.Client('localhost', 2000)
    client.set_timeout(120.0)
    world = client.load_world(settings['map'])
    vehicle_light_mask = set_weather(settings, world)

    create_vehicles(world, settings['num_vehicles'], vehicle_light_mask)
    create_people(world, settings['num_people'])

    time.sleep(3)

    actor, camera = create_actor(world, vehicle_light_mask)
  
    time.sleep(settings['duration_sec'])
    camera.destroy()
    actor.destroy()

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

def export_simulation(settings):
  movement_tracker = DashcamMovementTracker()
  movement_tracker.fps = OUTPUT_FRAMES_PER_SECOND
  video_time_sec = 0.0
  captured_frame_count = len(captured_frames)
  current_frame_index = 0
  stops = []
  current_stop = None
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
      video_time_sec += (1 / OUTPUT_FRAMES_PER_SECOND)
      if current_stop is None:
        if captured_moving[current_frame_index]:
          current_stop = captured_frame.timestamp * 1000
      else:
        if not captured_moving[current_frame_index]:
          stops.append((current_stop, captured_frame.timestamp * 1000))
          current_stop = None
    else:
      captured_frames[current_frame_index] = None
      current_frame_index += 1

    if current_frame_index + 1 >= captured_frame_count:
      break

  print(f'Processed {captured_frame_count} frames for {video_time_sec} seconds of video with {len(movement_tracker.frames)} frames')
  if current_stop is not None:
    stops.append((current_stop, None))
  movement_tracker.write_video(CONFIG.get_windows_temp_dir() + '/carla/' + str(settings['carla_id']) + '.mp4', include_timings=False)

  return stops




with psycopg2.connect(CONFIG.get_psycopg2_conn()) as db:
  while True:
    sql = 'SELECT carla_id, duration_sec, map, num_vehicles, num_people, weather_base, fog_density, fog_distance, precipitation, precipitation_deposits, wind_intensity, allocated FROM carla WHERE allocated = False LIMIT 1 FOR UPDATE'
    with db.cursor() as cursor:
      cursor.execute(sql)
      row = cursor.fetchone()

      if row is None:
        break
      sql = 'UPDATE carla SET allocated = True WHERE carla_id = ' + str(row[0])
      cursor.execute(sql)

      cursor.execute('commit')

      settings = {
        'carla_id': row[0],
        'duration_sec': row[1],
        'map': row[2],
        'num_vehicles': row[3],
        'num_people': row[4],
        'weather_base': row[5],
        'fog_density': row[6],
        'fog_distance': row[7],
        'precipitation': row[8],
        'precipitation_deposits': row[9],
        'wind_intensity': row[10],
        'allocated': row[11]
      }

      captured_frames = []
      captured_moving = []
      run_simulation(settings)
      stops = export_simulation(settings)
      
      for stop in stops:
        sql = 'INSERT INTO carla_stop(carla_id, stop_time_ms, start_time_ms) VALUES(' + str(settings['carla_id']) + ', ' + str(stop[0]) +', '
        if stop[1] is None:
          sql = sql + 'null'
        else:
          sql = sql + str(stop[1])
        sql = sql + ')'

        cursor.execute(sql)