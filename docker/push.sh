#!/bin/bash

IMAGENAME=kaggle-imc2024

docker tag ${IMAGENAME}:latest container-registry1.infrarist.com:5500/y_fujimoto/${IMAGENAME}:latest
docker push container-registry1.infrarist.com:5500/y_fujimoto/${IMAGENAME}:latest
