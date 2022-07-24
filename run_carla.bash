#!/bin/bash


for i in {1..40}
do
  python src/carla-gen.py
  sleep 5
done
