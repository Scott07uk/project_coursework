# Project Coursework for Richard Scott Neville (s5324494)

## Introduction

This project attempts to take video recorded on dash-cams, when the vehicle stops moving a prediction is made if the vehicle will start 
moving again within 8 seconds. Eight seconds is the time at which it becomes a fuel saving to stop the engine from running, any stop
less than 8 seconds, the extra fuel required to start the engine will be more than the fuel saved by stoping the engine.

The intention is to save fuel in vehicles with internal combusion engines by only activating any start-stop/Intelligence Stop & Go (ISG)
technology if a genuine fuel saving will be made.

## Requirements

The following is required to run this project

### Hardware
* A machine running Linux (target machine used was running OpenSuSE 15.3)
* At least 32GB RAM (more is better as training will occur faster, target machine had 64GB)
* At least 4 Core CPU (target machine was AMD Ryzen 5 3600X 6Cores/12Threads)
* Nvidia GPU with at least 12GB GDDR RAM (target machine used 3080Ti)

### Software
* Anaconda 3
  > bob

## Extracting Stop / Start Times From Real-World Data

## Generating Data With CARLA

## Extracting Data For Training

## Training Regression Models on Real-World Data

## Training Classification Models on Real-World Data

## Training Classification Models on Combined Data

## Testing / Understanding Models