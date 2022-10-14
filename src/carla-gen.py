import random
import carla
import time
from DashcamMovementTracker import DashcamMovementTracker
import cv2
import numpy
import re
from BDD import BDDConfig
import psycopg2
import psutil

CONFIG = BDDConfig('cfg/kastria-local.json')
ACTOR_CLASS = 'vehicle.ford.crown'
ANY_VEHICLE_CLASS = 'vehicle.*'
ANY_PERSON_CLASS = 'walker.pedestrian.*'
CAMERA_CLASS = 'sensor.camera.rgb'
CAM_FRAMES_PER_SECOND = 15
CAM_IMAGE_SIZE_X = 1280
CAM_IMAGE_SIZE_Y = 720
CAM_FIELD_OF_VIEW = 140
CAM_RELATIVE_LOCATION = carla.Location(x=2.2, y=0.0, z=1.1)
SPECTATOR_RELATIVE_LOCATION = carla.Location(x=0, y=0.0, z=80)
OUTPUT_FRAMES_PER_SECOND = 15
MOVEMENT_THRESH = 0.01

MEM_USAGE_CARLA = 1.5 * 1024.0 * 1024.0 * 1024.0
MEM_USAGE_OS = 2 * 1024.0 * 1024.0 * 1024.0
MEM_USAGE_PER_MIN = 5 * 1024 * 1024 * 1024


class CarlaSimulation:
  def __init__(self, settings):
    self.settings = settings
    self.captured_frames = []
    self.captured_moving = []
    self.running = False

  def record_video_frame(self, actor, image):
    if self.running:
      self.captured_frames.append(image)
      velocity = actor.get_velocity()
      self.captured_moving.append(abs(velocity.x) < MOVEMENT_THRESH and abs(velocity.y) < MOVEMENT_THRESH and abs(velocity.z) < MOVEMENT_THRESH)

  def create_actor(self, light_mask):
    spawn_points = self.world.get_map().get_spawn_points()
    while True:
      spawn_point = random.choice(spawn_points)
      if (spawn_point.rotation.yaw < 0.1) and (spawn_point.rotation.yaw > -0.1):
        break
    actor_bp = self.get_blueprint(ACTOR_CLASS)
    actor_bp.set_attribute('role_name', 'hero')
    camera_bp = self.get_blueprint(CAMERA_CLASS)
    camera_bp.set_attribute('sensor_tick', str(1 / CAM_FRAMES_PER_SECOND))
    camera_bp.set_attribute('image_size_x', str(CAM_IMAGE_SIZE_X))
    camera_bp.set_attribute('image_size_y', str(CAM_IMAGE_SIZE_Y))
    camera_bp.set_attribute('fov', str(CAM_FIELD_OF_VIEW))

    actor = None
    while actor is None:
      actor = self.world.try_spawn_actor(actor_bp, spawn_point)
      time.sleep(2)
  
    camera = self.world.spawn_actor(camera_bp, carla.Transform(CAM_RELATIVE_LOCATION), attach_to=actor)
  
    spectator = self.world.get_spectator()
    spectator.set_transform(carla.Transform(spawn_point.location + SPECTATOR_RELATIVE_LOCATION, carla.Rotation(pitch=-90)))

    self.traffic_manager.update_vehicle_lights(actor, True)
    actor.set_autopilot(True, self.traffic_manager.get_port())
    #actor.set_light_state(carla.VehicleLightState(light_mask))

    camera.listen(lambda image: self.record_video_frame(actor, image))
    return actor, camera

  def create_vehicles(self, light_mask):
    number = self.settings['num_vehicles']
    spawn_points = self.world.get_map().get_spawn_points()
    vehicles = []

    for index in range(number):
      spawn_point = random.choice(spawn_points)
      vehicle_bp = self.get_blueprint(ANY_VEHICLE_CLASS)
      vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
      if vehicle is not None:
        self.traffic_manager.update_vehicle_lights(vehicle, True)
        vehicle.set_autopilot(True, self.traffic_manager.get_port())
        #vehicle.set_light_state(carla.VehicleLightState(light_mask))
        vehicles.append(vehicle)
    return vehicles

  def create_people(self):
    number = self.settings['num_people']
    people = []
    for index in range(number):
      spawn_point = None
      while spawn_point is None:
        spawn_point = self.world.get_random_location_from_navigation()
    
      person_bp = self.get_blueprint(ANY_PERSON_CLASS)
      if person_bp.has_attribute('is_invincible'):
        person_bp.set_attribute('is_invincible', 'false')
      person = self.world.try_spawn_actor(person_bp, carla.Transform(spawn_point))
      if person is not None:
        people.append(person)

    return people

  def get_blueprint(self, blueprint):
    bp_list = self.world.get_blueprint_library().filter(blueprint)
    bp_list = random.choice(bp_list)
    return bp_list

  def set_weather(self):
    weather = getattr(carla.WeatherParameters, settings['weather_base'])
    if self.settings['fog_density'] is not None:
      weather.fog_density = self.settings['fog_density']
    if self.settings['fog_distance'] is not None:
      weather.fog_distance = self.settings['fog_distance']
    if self.settings['precipitation'] is not None:
      weather.precipitation = self.settings['precipitation']
    if self.settings['precipitation_deposits'] is not None:
      weather.precipitation_deposits = self.settings['precipitation_deposits']
    if self.settings['wind_intensity'] is not None:
      weather.wind_intensity = self.settings['wind_intensity']
    self.world.set_weather(weather)

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

  def run_simulation(self):
    camera = None
    actor = None
    client = None
    try:
      client = carla.Client('kastria.worldsofwar.co.uk', 2000)
      client.set_timeout(120.0)
      self.world = client.load_world(self.settings['map'])
      self.traffic_manager = client.get_trafficmanager(8000)
      vehicle_light_mask = self.set_weather()

      self.create_vehicles(vehicle_light_mask)
      self.create_people()

      time.sleep(3)
      self.running = True

      actor, camera = self.create_actor(vehicle_light_mask)
  
      time.sleep(self.settings['duration_sec'])
      self.running = False
      return True
    finally:
      if camera is not None:
        camera.destroy()
      if actor is not None:
        actor.destroy()
      if client is not None:
        client.stop_recorder()

    return False


  def to_bgra_array(self, image):
    array = numpy.frombuffer(image.raw_data, dtype=numpy.dtype("uint8"))
    array = numpy.reshape(array, (image.height, image.width, 4))
    return array

  def to_rgb_array(self, image):
    array = self.to_bgra_array(image)
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array

  def export_simulation(self):
    movement_tracker = DashcamMovementTracker()
    movement_tracker.fps = OUTPUT_FRAMES_PER_SECOND
    video_time_sec = 3.0
    captured_frame_count = len(self.captured_frames)
    
    current_frame_index = 0
    stops = []
    current_stop = None
    while True:
      this_frame_diff = abs(video_time_sec - self.captured_frames[current_frame_index].timestamp)
      next_frame_diff = abs(video_time_sec - self.captured_frames[current_frame_index + 1].timestamp)
      if this_frame_diff <= next_frame_diff:
        captured_frame = self.captured_frames[current_frame_index]
        reloaded_image = self.to_rgb_array(captured_frame)
        reloaded_image = cv2.cvtColor(reloaded_image, cv2.COLOR_RGB2BGR)
        movement_tracker.frames.append(reloaded_image)
        movement_tracker.frame_times.append(captured_frame.timestamp * 1000)  
        movement_tracker.frame_stop_status.append(self.captured_moving[current_frame_index])
        video_time_sec += (1 / OUTPUT_FRAMES_PER_SECOND)
        if current_stop is None:
          if self.captured_moving[current_frame_index]:
            current_stop = captured_frame.timestamp * 1000
        else:
          if not self.captured_moving[current_frame_index]:
            stops.append((current_stop, captured_frame.timestamp * 1000))
            current_stop = None
      else:
        self.captured_frames[current_frame_index] = None
        current_frame_index += 1

      if current_frame_index + 1 >= captured_frame_count:
        break

    print(f'Processed {captured_frame_count} frames for {video_time_sec} seconds of video with {len(movement_tracker.frames)} frames')
    if current_stop is not None:
      stops.append((current_stop, None))
    movement_tracker.write_video(CONFIG.get_temp_dir() + '/carla/' + str(settings['carla_id']) + '.mp4', include_timings=False)

    return stops



system_memory = psutil.virtual_memory().total
memory_for_simulation = system_memory - MEM_USAGE_OS - MEM_USAGE_CARLA
max_simulation_sec = (memory_for_simulation / MEM_USAGE_PER_MIN) * 60.0

print(f'This machine has enough memory to support a maximim simulation length of {max_simulation_sec} seconds')

with psycopg2.connect(CONFIG.get_psycopg2_conn()) as db:
  while True:
    sql = f'SELECT carla_id, duration_sec, map, num_vehicles, num_people, weather_base, fog_density, fog_distance, precipitation, precipitation_deposits, wind_intensity, allocated FROM carla WHERE allocated = False AND duration_sec <= {max_simulation_sec} ORDER BY duration_sec DESC LIMIT 1 FOR UPDATE'
    settings = {}
    with db.cursor() as cursor:
      cursor.execute(sql)
      row = cursor.fetchone()

      if row is None:
        print('No more simulations to process')
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

    print('Running simulation ' + str(settings['carla_id']))
    simulation = CarlaSimulation(settings)
    successful = simulation.run_simulation()
    print('Simulation ' + str(settings['carla_id']) + ' Complete, success = ' + str(successful))
    stops = simulation.export_simulation()
    
    with db.cursor() as cursor:
      for stop in stops:
        sql = 'INSERT INTO carla_stop(carla_id, stop_time_ms, start_time_ms) VALUES(' + str(settings['carla_id']) + ', ' + str(stop[0]) +', '
        if stop[1] is None:
          sql = sql + 'null'
        else:
          sql = sql + str(stop[1])
        sql = sql + ')'

        cursor.execute(sql)

    stops = None
    simulation = None

    exit()

print('Finishing')
