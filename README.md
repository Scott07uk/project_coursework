# Project Coursework for Richard Scott Neville (s5324494)

## Introduction

This project attempts to take video recorded on dash-cams, when the vehicle stops moving a prediction is made if the vehicle will start 
moving again within 8 seconds. Eight seconds is the time at which it becomes a fuel saving to stop the engine from running, any stop
less than 8 seconds, the extra fuel required to start the engine will be more than the fuel saved by stoping the engine.

The intention is to save fuel in vehicles with internal combusion engines by only activating any start-stop/Intelligence Stop & Go (ISG)
technology if a genuine fuel saving will be made.

This readme takes you through all the steps to run the project, alternativly you can jump straight to the last seconds to run the results
and see the visualisations.

## Requirements

The following is required to run this project

### Hardware
* A machine running Linux (target machine used was running OpenSuSE 15.3)
* At least 32GB RAM (more is better as training will occur faster, target machine had 64GB)
* At least 4 Core CPU (target machine was AMD Ryzen 5 3600X 6Cores/12Threads)
* Nvidia GPU with at least 12GB GDDR RAM (target machine used 3080Ti)

### Software
* Anaconda 3 (with the following modules installed)
  * bob
* ffmpeg (version 4.4 with ffprobe)
* Nvidia graphics card drives for your selected GPU
* Nvidia CUDA 11.4
* Postgres (any supported version)

### Setup
You will need to prime your postgres database with some of the relevant data, you can either:
* Create an empty database and generate your own data
* Prime the database with the existing data used within the project

To create an empty database run the following SQL files into your DB:
* sql/create_tables.sql
* sql/create_bdd_train.sql
* sql/insert_carla.sql

You will now have a database primed with the data for the CARLA simulation and the list of BDD videos.

To import the existing data run the follwing SQL file into your DB:
* sql/pg_dump.sql

You will need to ensure you have downloads the relevant carla-original files that go with this.

Processes that update the database are able to run on multiple machines at the same time, this will allow you to increase the amount of 
data you process by running the same process on multiple machines.

### Config

You will need to write a config file which has the details of where you want your temp data to be kept (this can be big and will be used extensivly during training) and also the location of your database. The default name of the config file is cfg/kastria-local.json (which is the name of the machine used for most training work).  You can edit the checked in file to fit your own setup.


## Extracting Stop / Start Times From Real-World Data

This process will need access to the raw BDD videos and the postgres database, it will lock one record on the database, process it then post its results back to the database. This means you can run multiple copies of this process on different machines to improve the throughput.

**WARNING**: This process is *VERY* slow

Run the following command:
```
python src/meta-gen-bdd.py
```

The process is already multi-threaded and will use multiple cores on a single machine, only a very limited performance gain will be achived by running multiple copies on the same machine (unless your machine has a very high core count)

## Generating Data With CARLA

There are two parts to this process:
* Running the CARLA Simulator (this is best run on a Windows 10 machine).
* Running the CARLA controller to run and record the simulation 

As with extracting the stop times this can be run on as many machines concurrently as you have, however you are not able to run multiple simulations concurrently within a single CARLA simulator, you must have multiple simulators to run concurrent simulations.

Run the following command:
```
python src/carla-gen.py
```
The following parameters can be added
--config <path-to-config-file>
--carla <carla-server>

For example
```
python src/carla-gen.py --config cfg/my-config-file.json --carla carla-desktop-1.my.fully.qualified.domain
```

## Extracting Data For Training

## Training Regression Models on Real-World Data

## Training Classification Models on Real-World Data

## Training Classification Models on Combined Data

## Testing / Understanding Models