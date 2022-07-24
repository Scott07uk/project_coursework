CREATE TABLE video_file (
  id        bigserial,
  dataset   varchar(10),
  file_type varchar(10),
  file_name varchar(100),
  state     varchar(10),
  PRIMARY KEY (id)
);


CREATE TABLE video_file_stop(
  video_file_id bigint REFERENCES video_file(id),
  stop_time_ms  integer,
  start_time_ms integer,
  PRIMARY KEY (video_file_id, stop_time_ms)
);


CREATE TABLE carla (
  carla_id               bigserial,
  duration_sec           integer,
  map                    varchar(20),
  num_vehicles           integer,
  num_people             integer,
  weather_base           varchar(20),
  fog_density            real,
  fog_distance           real, 
  precipitation          real,
  precipitation_deposits real,
  wind_intensity         real,
  allocated              boolean,
  PRIMARY KEY(carla_id)
);

CREATE TABLE carla_stop (
  stop_id       bigserial,
  carla_id      bigint    REFERENCES carla(carla_id),
  stop_time_ms  integer,
  start_time_ms integer,
  PRIMARY KEY(stop_id)
);
