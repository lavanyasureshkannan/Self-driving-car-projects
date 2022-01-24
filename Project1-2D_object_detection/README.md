
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

 ## DOWNLOAD DATA
For this project, we will be using data from the [Waymo Open dataset](https://waymo.com/open/). The files can be downloaded directly from the website as tar files or from the [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files) as individual tf records.

The datas downloaded will be in the form of waymo TF.record format. We need to convert the files into tensorflow API format - TF.record format. Therefore we are creating a function "create_tf_example" which takes in the waymo tf.record format and convert it into tensor flow object detection API format.
An example of such function is described [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#create-tensorflow-records).
We are also using `label_map.pbtxt file`.

You can run the script using the following (you will need to add your desired directory names):
```bash
python3 download_process.py --data_dir /home/lavanya/Downloads/project1nd/processed --temp_dir /home/lavanya/Downloads/project1nd/training_data

```
## EXPLORATORY DATA ANALYSIS

Now that you have downloaded and processed the data, you should explore the dataset! This is the most important task of any machine learning project. To do so, open the Exploratory Data Analysis notebook.  `Display_instances` function is implemented to display images and annotations using matplotlib

## EDIT THE CONFIG FILE
Now you are ready for training. The Tf Object Detection API relies on config files. The config that we will use for this project is pipeline.config, which is the config for a SSD Resnet 50 640x640 model. You can learn more about the Single Shot Detector [here](https://arxiv.org/pdf/1512.02325.pdf).
First, let's download the [pretrained model](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) and move it to training/pretrained-models/.

Now we need to edit the config files to change the location of the training and validation files, as well as the location of the label_map file, pretrained weights. We also need to adjust the batch size. MOdification to be made,

```bash
num_classes: number of classes
batch_size: according to your computer memory
fine_tune_checkpoint: modify to pretrained model ckpt-0 path
num_steps: training steps
use_bfloat16: whether to use tpu, if not used, set to false
label_map_path: label_map.pbtxt path
train_input_reader: set input_path to the tfrecord path for training
metrics_set: “coco_detection_metrics”
use_moving_averages: false
eval_input_reader: Set input_path to the tfrecord path for evaluating
```
## TRAINING 
A new config file has been created, `pipeline_new.config`.
Create a folder training/reference. Move the pipeline_new.config to this folder. You will now have to launch,
```bash
python3 model_main_tf2.py --model_dir=/home/lavanya/Downloads/project1nd/training/reference/ --pipeline_config_path=/home/lavanya/Downloads/project1nd/training/reference/pipeline_new.config
```
## EXPORT THE MODEL
Export the trained model
```bash
python3 exporter_main_v2.py --input_type image_tensor --pipeline_config_path /home/lavanya/Downloads/project1nd/training/reference/pipeline_new.config --trained_checkpoint_dir /home/lavanya/Downloads/project1nd/training/reference/ ckpt-26 --output_directory /home/lavanya/Downloads/project1nd/training/reference/experiment/exported_model_26/
```
## TEST THE MODEL
Test the model for each frame.
Created a new file `detect.py` for testing the images. Run the following code based on your directory path.
```bash
python3 detect.py --saved_model_path=/home/lavanya/Downloads/project1nd/training/reference/experiment/exported_model/saved_model --test_path=/home/lavanya/Downloads/project1nd/test_image --output_path=/home/lavanya/Downloads/project1nd/output/ --min_score_thresh=.1
```
## REFERENCES
1. [Self driving car- udacity Nano degree](https://classroom.udacity.com/nanodegrees/nd0013/parts/6f5460e5-fab9-4c78-b31a-ea5e7f1d285c/modules/52716294-bc19-4cd7-b332-64e5250cd613/lessons/39c92933-f5ba-4b80-963e-b78f50ef45a6/concepts/c4a046bc-3653-4e01-84f9-b2c4a67d610c)
2. [Tensorflow object detection API tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/index.html)
3. [waymo open dataset - quick start tutorial](https://colab.research.google.com/github/waymo-research/waymo-open-dataset/blob/r1.0/tutorial/tutorial.ipynb)
4. [gyes-object detection api](https://gyes00205.medium.com/waymo-open-dataset-tensorflow-2-object-detection-api-49e38671f69f)
5. [Onject detction api YOUTUBE turotial - sentdex](https://www.youtube.com/watch?v=COlbP62-B-U&list=PLQVvvaa0QuDcNK5GeCQnxYnSSaar2tpku)

