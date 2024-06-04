FROM python:3.10.13-slim AS base
# FROM python:3.10.10-slim AS base  # work
# FROM python:3.10.12-slim AS base  # not work

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES utility,compute
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libxrender-dev build-essential unzip libvips-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# https://pytorch.org/get-started/previous-versions/
RUN pip install --upgrade pip \
    && pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir kaggle

RUN pip install --upgrade pip \
    && pip install --no-cache-dir opencv-contrib-python==4.9.0.80 \
    && pip install --no-cache-dir opencv-python==4.9.0.80 \
    && pip install --no-cache-dir h5py==3.10.0 \
    && pip install --no-cache-dir pandas==2.1.4 \
    && pip install --no-cache-dir transformers==4.38.1 \
    && pip install --no-cache-dir click==8.1.7

# h5py-3.11.0

COPY imc2024-packages-pycolmap-lightglue-rerun-kornia/ /root/imc2024-packages
RUN pip install --no-index /root/imc2024-packages/* --no-deps

# Language settings
ENV LANG C.UTF-8
ENV LANGUAGE en_US

WORKDIR workspace
