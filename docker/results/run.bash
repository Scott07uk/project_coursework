#!/bin/bash

mkdir /mnt/results

if [[ -z "${MODEL}" ]]; then
  echo "Environment Variable MODEL is not set, going to assume 1"
  MODEL="1"
fi

echo "Going to test model ${MODEL}"
ARCH="resnet50"

case "${MODEL}" in
  "1" | "2" | "3")
    ARCH="resnet50"
    ;;
  *)
    echo "${MODEL} is not a valid model"
    exit
esac

CAM_ARG=""

case "${CAM}" in 
  "ScoreCAM" | "GradCAM" | "SmoothGradCAMpp" | "GradCAMpp")
    CAM_ARG="--cam ${CAM}"
    mkdir /mnt/results/$CAM
    ln -s /mnt/results/$CAM /usr/local/project/data/$CAM
    ;;
esac

if [[ -z "${FUEL_ML_PER_SEC}" ]]; then
  FUEL_ML_PER_SEC=0.2020
fi

if [[ -z "${FUEL_START}" ]]; then
  FUEL_START=1.616
fi

python3 src/bdd-test.py --config cfg/docker.json --model models/$MODEL.cktp --arch $ARCH --csv --images still --test-set-file models/bdd-test-set.csv ${CAM_ARG} > /mnt/results/results.csv

echo "Results from test"
echo ""
cat /mnt/results/results.csv
echo ""
echo ""
echo ""
echo ""
echo ""
echo "---------------------------------------------------------------------------------------------------"
echo ""

python3 src/bdd-valid-info.py  --config cfg/docker.json --idle-fuel-use $FUEL_ML_PER_SEC --start-fuel-use $FUEL_START --results-csv /mnt/results/results.csv --stop results
python3 src/bdd-valid-info.py  --config cfg/docker.json --idle-fuel-use $FUEL_ML_PER_SEC --start-fuel-use $FUEL_START --results-csv /mnt/results/results.csv --stop never
python3 src/bdd-valid-info.py  --config cfg/docker.json --idle-fuel-use $FUEL_ML_PER_SEC --start-fuel-use $FUEL_START --results-csv /mnt/results/results.csv --stop always