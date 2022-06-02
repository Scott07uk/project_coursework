from BDD import BDDConfig, json_file_to_bdd_video, run_function_in_parallel
from DashcamMovementTracker import DashcamMovementTracker
import os
import threading
import uuid
import json


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

PROCESSED_FILE = 'cfg/meta-data-gen.json'

CONFIG = BDDConfig('cfg/laptop.json')
WRITE_ON_UPDATES = 1

processed_files = {'processed_videos': []}

if os.path.exists(PROCESSED_FILE):
  with open(PROCESSED_FILE) as processed_file_content:
    processed_files = json.load(processed_file_content)


info_files_to_load = []

file_limit_prefix = CONFIG.get_file_limit_prefix()

for file_type in CONFIG.get_types_to_load():
  for file in CONFIG.get_info_dir_ls(file_type):
    if file_limit_prefix is None or file.name.startswith(file_limit_prefix):
      if file.name.replace('json', 'mov') not in processed_files['processed_videos']:
        info_files_to_load.append((file_type, file))
  
    if (len(info_files_to_load) >= CONFIG.get_max_files_to_read()):
      break;
  if (len(info_files_to_load) >= CONFIG.get_max_files_to_read()):
    break;

print(f'Going to process {len(info_files_to_load)} file')

# Lets not do any threading
for info_file in info_files_to_load:
  print(f'Processing {info_file}')
  process_info_file(info_file)

#print('end')
#write_processed_files()
