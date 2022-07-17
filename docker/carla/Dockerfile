FROM opensuse/leap:15

RUN zypper install -y python3
RUN zypper install -y python3-tqdm python3-ffmpeg-python python3-opencv
RUN zypper install -y python3-numpy ffmpeg python3-pip
RUN zypper install -y libgthread-2_0-0
RUN zypper install -y python3-psycopg2 postgresql14-server-devel
RUN zypper install -y gcc python3-devel
RUN pip3 install opencv-contrib-python carla

WORKDIR /usr/local/carla

ADD src /usr/local/carla/src
ADD cfg /usr/local/carla/

CMD python3 src/carla-gen.py
