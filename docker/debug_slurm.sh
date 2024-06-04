#!/bin/bash

registry=container-registry1.infrarist.com:5500
image=y_fujimoto/kaggle-imc2024:latest

srun --container-image=${registry}#${image} \
    --container-mounts=$PWD:/workspace,/mnt/kaggle-y_fujimoto-data:/mnt/kaggle-y_fujimoto-data,$PWD/cache:/root/.cache \
    --qos=kaggle \
    --partition=kaggle-y-fujimoto \
    --nodes=1 \
    --ntasks=1 \
    --gres=gpu:2 \
    --mem=100Gb \
    --time=240:00:00 \
    --no-container-mount-home \
    --pty /bin/bash

#    --partition=rist20 \

