#!/bin/bash

IMAGENAME=kaggle-imc2024
DIRNAME=docker

docker build -t ${IMAGENAME} -f ${DIRNAME}/Dockerfile ./${DIRNAME}
