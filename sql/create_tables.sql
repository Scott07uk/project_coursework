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
