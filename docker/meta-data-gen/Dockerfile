FROM opensuse/leap:15

RUN zypper install -y python3
RUN zypper install -y python3-tqdm python3-ffmpeg-python python3-opencv
RUN zypper install -y python3-numpy ffmpeg python3-pip
RUN zypper install -y libgthread-2_0-0
RUN pip3 install opencv-contrib-python postgres

WORKDIR /usr/local/meta-data-gen

ADD src /usr/local/meta-data-gen/src
ADD cfg /usr/local/meta-data-gen/

CMD python3 src/meta-gen-bdd.py
