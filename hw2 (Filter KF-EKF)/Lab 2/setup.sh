#!/usr/bin/env bash
set -euxo pipefail

sudo apt-get update
sudo apt-get install -y \
      python3-catkin-tools \
      python3-pytest \
      python3-skimage \
      python3-pip \
      ros-noetic-rviz \
      ros-noetic-tf2-ros \
      ros-noetic-map-server

# Install Python packages with pip3
pip3 install -U numpy scikit-image numba matplotlib
