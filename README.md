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

    python src/meta-gen-bdd.py


The process is already multi-threaded and will use multiple cores on a single machine, only a very limited performance gain will be achived by running multiple copies on the same machine (unless your machine has a very high core count)

## Generating Data With CARLA

There are two parts to this process:

 * Running the CARLA Simulator (this is best run on a Windows 10 machine).
 * Running the CARLA controller to run and record the simulation 

As with extracting the stop times this can be run on as many machines concurrently as you have, however you are not able to run multiple simulations concurrently within a single CARLA simulator, you must have multiple simulators to run concurrent simulations.

Run the following command:

    python src/carla-gen.py

The following parameters can be added

    --config \<path-to-config-file\>
    --carla \<carla-server\>

For example

    python src/carla-gen.py --config cfg/my-config-file.json --carla carla-desktop-1.my.fully.qualified.domain


Video files will be written to a carla-orig directory within the temp directory defined in your configuration. The stop times will be written to the carla_stop table within the PostgreSQL database.  All the video files will need to be combined together for further processing (note that the file names will be based on the unique IDs generated by the database and therefore two nodes processing data concurrently will not generate conflicting files).

## Extracting Data For Training

There are two parts to this process:

 * Extracting real world data from BDD
 * Extracting synthetic data from the CARLA videos

There are two different processes to do this, but they both use the same functions inside DashcamMomentTracker to extract the training data.  Data will be extracted into:

 * \<TYPE\>-still - single still at the moment of the stop
 * \<TYPE\>-multi-still - Single image using three different colour channels to represent moment of stop, 2 seconds before stop, 4 seconds before stop
 * \<TYPE\>-video - Six second video preceeding the stop

\<TYPE\> will be replaced by the type (eg BDD or CARLA).  The format of the file names is the same for all videos which is:

 * \<IDENTIFIER\>-\<STOP_TIME\>.mp4 - for video files
 * \<IDENTIFIER\>-\<STOP_TIME\>/\<INDEX\>.jpeg - for image files

CARLA and BDD use different identifiers, CARLA uses the stop ID which the database allocated, BDD the file name of the video. INDEX's for images are all the same, with 19 being the frame at the moment of the stop 18 one frame before, 17 2 frames before etc.

Therefore the following filenames can be taken as examples:

 * bdd-video/02bb67ae-8c3d61f8.mov-14363.0.mp4 - BDD video file 02bb67ae-8c3d61f8.mov where the stop is 14363.0 milliseconds into the video
 * bdd-multi-still/02cdc06d-5502f174.mov-22542.0/19.jpeg - BDD image from 02cdc06d-5502f174.mov for a stop that occured 22542.0 milliseconds into the video at the moment the stop occurred
 * carla-still/988-54.234/9.jpeg - CARLA image for stop ID 988 which occurred 54.234 seconds into the video, 10 frames back from the moment of stop.

NOTE: CARLA videos are timed in seconds and BDD videos are times in milliseconds, this was an accidental oversight and could be corrected in the future.

### Extracting BDD Data
BDD Data is extracting using the following python script:

    src/bdd-extract.py

There are a number of parameters that you can provide to the script to control how the extract works

    --config <config-file> - Defaults to cfg/kastria-local.json
    --perform-extract - Extracts data from the source videos
    --process-all - If this is set then the videos will always be processed, if not set, it will only process missiong ones
    --dense-optical-flow - Extract the dense optical flow files
    --sparse-optical-flow - Extract the sparse optical flow files
    --train-sparse-optical-flow - Train a classifier for long/short stops using the sparse optical flow files
    --train-dense-optical-flow - Train a classifier for long/short stops using the dense optical flow files
    --arch - Either resnet50 or densenet121 (default). Architecture to use for the sparse/dense optical flow models

Data is extracted to the tempDir listed in the config file

### Extracting CARLA data
CARLA data is extracted (from the raw generated videos) using the following python script (note this script is also used for training synthetic data and combined models):

    src/carla-extract.py

There are a number of parameters that you can provide to the script to control how the extract works:

    --perform-extract - Set this to run the extract
    --dense-optical-flow - Set to extract the dense optical flow data
    --sparse-optical-flow - Set to extract the sparse optical flow data
    --perform-carla-mods - Perform random augmentations on the images before outputing (blur / lighting / contrast)
    --perform-stop-start-extract - Extract the frames which are stationary / moving for training stationary / moving models

NOTE: There are further parameters for carla-extract.py which are used for training models, these are explained in the training sections.


## Training Regression Models on Real-World Data

Regression models are training using four different python scripts:

* src/stop-time-trainer-stills.py - Trains using the single still at the moment of stop using Resnet50 / Densenet121 / EfficientNet B7
* src/stop-time-trainer-multi-stills.py - Trains using the three channel still at the moment of stop using Resnet50 / Densenet121 / EfficientNet B7
* src/stop-time-trainer-video.py - Trains a video model using a 4D based resnet (without pre-trained weights)
* src/stop-time-trainer-video-pre-trained.py - Trains using extracted videos using Slowfast with pre-trained weights.

For the scripts which can train with different architectures, slight changes to the code are required to switch between the architectures.  For example:

    #BATCH_SIZE = 16 #Resnet50
    #BATCH_SIZE = 12 #Densenet121
    BATCH_SIZE = 3 #EfficientNetB7

    # Resnet 50
    #self.model = models.resnet50(pretrained=True) 
    #self.model.fc = nn.Linear(in_features=2048, out_features=1)

    #Densenet121
    #self.model = models.densenet121(pretrained=True) 
    #self.model.classifier = nn.Linear(in_features=1024, out_features=1)

    #EfficientNetB7
    self.model = models.efficientnet_b7(pretrained=True)
    self.model.classifier[1] = nn.Linear(in_features=2560, out_features=1)

This code is setup to use the EfficientNetB7 architecture. To change this to use Resnet50, the following changes should be made:

    BATCH_SIZE = 16 #Resnet50
    #BATCH_SIZE = 12 #Densenet121
    #BATCH_SIZE = 3 #EfficientNetB7

    # Resnet 50
    self.model = models.resnet50(pretrained=True) 
    self.model.fc = nn.Linear(in_features=2048, out_features=1)

    #Densenet121
    #self.model = models.densenet121(pretrained=True) 
    #self.model.classifier = nn.Linear(in_features=1024, out_features=1)

    #EfficientNetB7
    #self.model = models.efficientnet_b7(pretrained=True)
    #self.model.classifier[1] = nn.Linear(in_features=2560, out_features=1)


## Training Classification Models on Real-World Data

Classification models are essentially the same as the regression models with the exception that they output two classes rather than a continious output. Like the regression models there are 4 different scripts:

* src/stop-time-trainer-stills-classifier.py - Trains using the single still at the moment of stop using Resnet50 / Densenet121 / EfficientNet B7
* src/stop-time-trainer-multi-stills-classifier.py - Trains using the three channel still at the moment of stop using Resnet50 / Densenet121 / EfficientNet B7
* src/stop-time-trainer-video-classifier.py - Trains a video model using a 4D based resnet (without pre-trained weights)
* src/stop-time-trainer-video-pre-trained-classifier.py - Trains using extracted videos using Slowfast with pre-trained weights.

The same notes apply regarding changes to the code as per the regression models.

## Training Classification Models on Combined Data

## Testing / Understanding Models