# Semantic Segmentation

The dataset that used for our experiments is the [M2NIST Dataset](https://www.kaggle.com/farhanhubble/multimnistm2nist). The dataset was created in order to teach the basics of semantic segmentation with convolutional neural networks without requiring the use of complex architechtures that take long to train. More details can be found at the above link.

## Installation and Required Packages

Matlab 2020a

Toolboxes utilized:
- Deep Learning Toolbox
- Computer Vision Toolbox
- Parallel Computing Toolbox

Python Packages: 
- keras
- Numpy
- OpenCV
- pathlib

If you don't have the above python packages. They can be installed using pip. 

```
$ pip install numpy
$ pip install opencv-python
$ pip install pathlib
$ pip install keras
```

## Setup 
The experiments located in this respository were conducted using a machine with the following specifications:
```
OS: Ubuntu 16.04.6 LTS (Xenial Xerus), 4.15.0-45-generic x86_64
GPU(s): 1 GeForce GTX 1080
```
To reproduce the results:

### Generate the Datasets 
1. Download the M2NIST Dataset .npy files from the following [link](https://www.kaggle.com/farhanhubble/multimnistm2nist) and place them in the dataset folder.
2. Run setup.sh. 
```
chmod u+x setup.sh && ./setup.sh
```

### Training Models

The training scripts used to train the segmentation models are located in the [matlab](matlab) directory. Execute each of them to train segmentation models on the datasets generated above. 



