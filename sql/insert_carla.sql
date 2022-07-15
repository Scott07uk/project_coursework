-- Weather Base:
-- 'ClearNight', 'ClearNoon', 'ClearSunset', 'CloudyNight', 'CloudyNoon', 'CloudySunset', 'Default', 'HardRainNight', 'HardRainNoon', 'HardRainSunset', 'MidRainSunset', 'MidRainyNight', 'MidRainyNoon', 'SoftRainNight', 'SoftRainNoon', 'SoftRainSunset', 'WetCloudyNight', 'WetCloudyNoon', 'WetCloudySunset', 'WetNight', 'WetNoon', 'WetSunset'

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, precipitation,
                  precipitation_deposits, wind_intensity, allocated)
VALUES(60, 'Town01', 10, 10,
       'Default', 0, 0, 0,
       0, 0, False);

INSERT INTO carla(duration_sec, map, num_vehicles, num_people, 
                  weather_base, fog_density, fog_distance, precipitation,
                  precipitation_deposits, wind_intensity, allocated)
VALUES(300, 'Town01', 10, 10,
       'ClearNoon', 0, 0, 0,
       0, 0, False);