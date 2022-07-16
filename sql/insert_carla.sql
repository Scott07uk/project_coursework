-- Weather Base:
-- 'ClearNight', 'ClearNoon', 'ClearSunset', 'CloudyNight', 'CloudyNoon', 'CloudySunset', 'Default', 'HardRainNight', 'HardRainNoon', 'HardRainSunset', 'MidRainSunset', 'MidRainyNight', 'MidRainyNoon', 'SoftRainNight', 'SoftRainNoon', 'SoftRainSunset', 'WetCloudyNight', 'WetCloudyNoon', 'WetCloudySunset', 'WetNight', 'WetNoon', 'WetSunset'

DELETE FROM carla_stop;
DELETE FROM carla;

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
VALUES(300, 'Town01', 3, 1, 'ClearNight', 0, 0, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 3, 1, 'ClearNight', 30, 60, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 3, 1, 'ClearNight', 80, 20, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 3, 1, 'CloudyNight', 0, 0, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 3, 1, 'CloudyNight', 30, 60, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 3, 1, 'HardRainNight', 0, 0, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 3, 1, 'MidRainyNight', 0, 0, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 3, 1, 'SoftRainNight', 0, 0, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 3, 1, 'WetNight', 0, 0, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 3, 1, 'WetNight', 30, 60, 0, False);

-- Night rush hour
INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 60, 80, 'ClearNight', 0, 0, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 60, 80, 'ClearNight', 30, 60, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 60, 80, 'ClearNight', 80, 20, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 60, 80, 'CloudyNight', 0, 0, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 60, 80, 'CloudyNight', 30, 60, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 60, 80, 'HardRainNight', 0, 0, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 60, 80, 'MidRainyNight', 0, 0, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 60, 80, 'SoftRainNight', 0, 0, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 60, 80, 'WetNight', 0, 0, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 60, 80, 'WetNight', 30, 60, 0, False);


UPDATE carla SET duration_sec = 30;