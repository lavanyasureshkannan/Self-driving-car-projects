
# INSTRUCTIONS

# Docker Setup

For local setup if you have your own Nvidia GPU, you can use the provided Dockerfile and requirements in the build directory of the starter code.

The instructions below are also contained within the build directory of the starter code.

Requirements

1. NVIDIA GPU with the latest driver installed

2. docker / nvidia-docker





## Build
Build the image with:


```bash
  docker build -t project-dev -f Dockerfile .

```
```bash
  docker build -t project-dev -f Dockerfile .
```
Create a container with:
```bash
  docker run --gpus all -v <PATH TO LOCAL PROJECT FOLDER>:/app/project/ --network=host -ti project-dev bash

```
and any other flag you find useful to your system (eg, --shm-size).

# Set up
Once in container, you will need to install gsutil, which you can easily do by running:
```bash
  curl https://sdk.cloud.google.com | bash

```
Once gsutil is installed and added to your path, you can auth using:
```bash
  gcloud auth login

```
Debug
Follow this [tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tensorflow-object-detection-api-installation)
 if you run into any issue with the installation of the TF Object Detection API.






