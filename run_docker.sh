#!/bin/bash

#xhost +local:docker

docker run --gpus all -it --rm -e DISPLAY=$DISPLAY  -v /tmp/.X11-unix:/tmp/.X11-unix:ro -p 5901:5901 -p 6080:6080 --shm-size=512M  matlab_semantic_segmentation:0.0.1