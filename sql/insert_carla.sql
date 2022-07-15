-- Weather Base:
-- 'ClearNight', 'ClearNoon', 'ClearSunset', 'CloudyNight', 'CloudyNoon', 'CloudySunset', 'Default', 'HardRainNight', 'HardRainNoon', 'HardRainSunset', 'MidRainSunset', 'MidRainyNight', 'MidRainyNoon', 'SoftRainNight', 'SoftRainNoon', 'SoftRainSunset', 'WetCloudyNight', 'WetCloudyNoon', 'WetCloudySunset', 'WetNight', 'WetNoon', 'WetSunset'

--
-- Situations to cover for each map
-- Time of Day:
--   Night - quite (ie overnight)
--   Night - rush hour - very busy
--   Noon - moderate busy
--   Sunset - quite late at night
--   Sunset - rush hour
--
-- Weather:
--  Clear - no fog
--  Clear - light fog
--  Clear - heavy fog
--  Cloudy - no fog
--  Cloudy - light fog
--  HardRain - no fog
--  MidRain - no fog
--  SoftRain - no fog
--  Wet - no fog
--  Wet - light fog

-- Night quiet
INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(180, 'Town01', 3, 1, 'ClearNight', 0, 0, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(180, 'Town01', 3, 1, 'ClearNight', 30, 60, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(180, 'Town01', 3, 1, 'ClearNight', 80, 20, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(180, 'Town01', 3, 1, 'CloudyNight', 0, 0, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(180, 'Town01', 3, 1, 'CloudyNight', 30, 60, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(180, 'Town01', 3, 1, 'HardRainNight', 0, 0, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(180, 'Town01', 3, 1, 'MidRainNight', 0, 0, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(180, 'Town01', 3, 1, 'SoftRainNight', 0, 0, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(180, 'Town01', 3, 1, 'WetNight', 0, 0, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(180, 'Town01', 3, 1, 'WetNight', 30, 60, 0, False);

-- Night rush hour
INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(180, 'Town01', 20, 30, 'ClearNight', 0, 0, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(180, 'Town01', 20, 30, 'ClearNight', 30, 60, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(180, 'Town01', 20, 30, 'ClearNight', 80, 20, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(180, 'Town01', 20, 30, 'CloudyNight', 0, 0, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(180, 'Town01', 20, 30, 'CloudyNight', 30, 60, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(180, 'Town01', 20, 30, 'HardRainNight', 0, 0, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(180, 'Town01', 20, 30, 'MidRainNight', 0, 0, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(180, 'Town01', 20, 30, 'SoftRainNight', 0, 0, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(180, 'Town01', 20, 30, 'WetNight', 0, 0, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(180, 'Town01', 20, 30, 'WetNight', 30, 60, 0, False);