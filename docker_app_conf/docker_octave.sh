#!/usr/bin/env bash
set -xe
xhost +local:

docker run --rm -it -e DISPLAY=$DISPLAY -e LIBGL_ALWAYS_SOFTWARE=1 -v /home/hpsilveira/Documentos/ML:/usr/local/ml -v /tmp/.X11-unix:/tmp/.X11-unix mtmiller/octave:6.0.1
