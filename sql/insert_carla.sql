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
VALUES(300, 'Town01', 80, 100, 'ClearNight', 0, 0, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 80, 100, 'ClearNight', 30, 60, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 80, 100, 'ClearNight', 80, 20, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 80, 100, 'CloudyNight', 0, 0, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 80, 100, 'CloudyNight', 30, 60, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 80, 100, 'HardRainNight', 0, 0, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 80, 100, 'MidRainyNight', 0, 0, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 80, 100, 'SoftRainNight', 0, 0, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 80, 100, 'WetNight', 0, 0, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 80, 100, 'WetNight', 30, 60, 0, False);

-- Noon
INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 40, 50, 'ClearNoon', 0, 0, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 40, 50, 'ClearNoon', 30, 60, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 40, 50, 'ClearNoon', 80, 20, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 40, 50, 'CloudyNoon', 0, 0, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 40, 50, 'CloudyNoon', 30, 60, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 40, 50, 'HardRainNoon', 0, 0, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 40, 50, 'MidRainyNoon', 0, 0, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 40, 50, 'SoftRainNoon', 0, 0, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 40, 50, 'WetNoon', 0, 0, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 40, 50, 'WetNoon', 30, 60, 0, False);


-- Sunset Quite
INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 20, 40, 'ClearSunset', 0, 0, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 20, 40, 'ClearSunset', 30, 60, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 20, 40, 'ClearSunset', 80, 20, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 20, 40, 'CloudySunset', 0, 0, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 20, 40, 'CloudySunset', 30, 60, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 20, 40, 'HardRainSunset', 0, 0, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 20, 40, 'MidRainSunset', 0, 0, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 20, 40, 'SoftRainSunset', 0, 0, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 20, 40, 'WetSunset', 0, 0, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, wind_intensity, allocated)
VALUES(300, 'Town01', 20, 40, 'WetNight', 30, 60, 0, False);

UPDATE carla SET duration_sec = 30;
