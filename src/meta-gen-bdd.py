from BDD import BDDConfig, json_file_to_bdd_video, run_function_in_parallel
from DashcamMovementTracker import DashcamMovementTracker
import os
import threading
import uuid
import json
from postgres import Postgres


def process_info_file(info_file):
  file_type, file_name = info_file
  bdd_video = json_file_to_bdd_video(CONFIG, file_type, file_name)
  movement_tracker = DashcamMovementTracker()
  stops = movement_tracker.get_stops_from_file(bdd_video.get_absoloute_file_name())
  
  if not stops is None:
    for stop in stops:
      bdd_video.record_raw_stop_time('movement', stop[0], stop[1])

    output_filename = CONFIG.get_meta_data_dir() + '/' + str(uuid.uuid4()) + '.json'
    bdd_metadata = bdd_video.to_metadata()
    with open(output_filename, "w") as output_file:
      output_file.write(json.dumps(bdd_metadata, indent = 2))
  file_processed(bdd_video.get_file_name())

def file_processed(file_name):
  processed_files['processed_videos'].append(file_name)
  write_processed_files(processed_files)

def write_processed_files(processed_files):
  json_str = json.dumps(processed_files, indent = 2)
  with open(PROCESSED_FILE, "w") as outfile:
    outfile.write(json_str)

CONFIG = BDDConfig('cfg/laptop.json')

db = Postgres(url='postgres://project:password@oseidon.worldsofwar.co.uk/dev')

row = db.one('SELECT * FROM video_file WHERE state = \'NEW\' LIMIT 1 FOR UPDATE', back_as='dict')

while row is not None:
  id = row['id']
  dataset = row['dataset']
  file_type = row['file_type']
  file_name = row['file_name']

  db.run('UPDATE video_file SET state = %s WHERE id=%s', ('PENDING', id))

  absoloute_file_name = CONFIG.get_absoloute_path_of_video(file_type, file_name)

  movement_tracker = DashcamMovementTracker()
  stops = movement_tracker.get_stops_from_file(absoloute_file_name)
  
  db.run(f'DELETE FROM video_file_stop WHERE video_file_id = {id}')

  if not stops is None:
    for stop in stops:
      db.run('INSERT INTO video_file_stop(video_file_id, stop_time_ms, start_time_ms) VALUES(%s, %s, %s)', (id, stop[0], stop[1]))

  db.run('UPDATE video_file SET state = %s WHERE id=%s', ('DONE', id))

  row = db.one('SELECT * FROM video_file WHERE state = \'NEW\' LIMIT 1 FOR UPDATE', back_as='dict')