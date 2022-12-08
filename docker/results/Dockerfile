FROM ubuntu:22.04

RUN apt-get update
RUN apt-get install -y python3 python3-pip ffmpeg bzip2

RUN pip3 install opencv-contrib-python postgres pillow pytorch_lightning torchvision ffmpeg-python pytorchvideo pandas

WORKDIR /usr/local/project

ADD data/bdd-still.tar.bz2 /usr/local/project/data/
ADD data/bdd-multi-still.tar.bz2 /usr/local/project/data/

ADD m/bdd-test-set.csv /usr/local/project/models/bdd-test-set.csv

ADD m/1.ckpt /usr/local/project/models/1.cktp

ADD src /usr/local/project/src
ADD cfg /usr/local/project/cfg
ADD docker/results/run.bash /usr/local/project/

CMD /bin/bash run.bash
