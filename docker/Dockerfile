FROM nvcr.io/nvidia/pytorch:23.03-py3

RUN apt-get update && apt-get -y upgrade

ENV TZ=Europe/London
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
     lsof \
     wget  \
     bzip2  \
     libglib2.0-0 \
     libxext6 \
     libsm6 \
     libxrender1 \
     build-essential \
     cmake \
     git \
     curl \
     vim \
     ca-certificates \
     libjpeg-dev \
     ssh \
     htop \
     apt-transport-https \
     libpng-dev \
     gnupg \
     yasm \
     ffmpeg  \
     zip \
     tmux &&\
   rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --upgrade setuptools

COPY requirements.txt requirements.txt
RUN rm -r /usr/local/lib/python3.8/dist-packages/cv2
RUN python -m pip install -r requirements.txt

WORKDIR /workdir
