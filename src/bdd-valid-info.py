from argparse import (
  ArgumentParser
)
from BDD import (
  BDDConfig, 
  video_stops_from_database
)
import pandas

parser = ArgumentParser()

parser.add_argument('--config', dest='config', action='store', help='The config file to use')
parser.add_argument('--idle-fuel-use', dest='idle_fuel_use', action='store', help='The fuel use for ideling')
parser.add_argument('--start-fuel-use', dest='start_fuel_use', action='store', help='The fuel used in starting the engine')
parser.add_argument('--results-csv', dest='results_csv', action='store', help='CSV of the results to use')
parser.add_argument('--stop', dest='stop', action='store', help='Either always, never or results')

parser.set_defaults(config = 'cfg/kastria-local.json', stop = 'never')
args = parser.parse_args()

CONFIG = BDDConfig(args.config)

video_train, video_test = video_stops_from_database(CONFIG)

stops = []

if (args.stop == 'never'):
  stops = [False] * len(video_test)
elif (args.stop == 'always'):
  stops = [True] * len(video_test)
elif (args.stop == 'results'):
  data_frame = pandas.read_csv(args.results_csv)
  for video in video_test:
    file_name = video['file_name']
    row = data_frame.loc[data_frame['FileName'] == file_name]
    predicted = row['Predicted']
    stops.append(predicted == 1)
else:
  print('Invalid arguments, stop must be [never, always, results]')
  exit()

total_fuel_cost = 0.0

for index in range(len(video_test)):
  video = video_test[index]
  stop = stops[index]
  if (args.stop == 'results'):
    stop = stop.bool()

  if (stop):
    total_fuel_cost = total_fuel_cost + float(args.start_fuel_use)
  else:
    total_fuel_cost = total_fuel_cost + (float(args.idle_fuel_use) * (video['duration'] / 1000.0))

  total_fuel_cost = round(total_fuel_cost, 5)


print('+-------------+')
print('|   RESULTS   |')
print('+-------------+')
print('')
print(f'Stopping: {args.stop}')
print(f'Fuel Cost per second of idle (ml): {args.idle_fuel_use}')
print(f'Fuel Cost engine start (ml): {args.start_fuel_use}')
print(f'Test data {len(video_test)}')
print(f'Fuel Usage: {total_fuel_cost} ml')
