# Semantic Segmentation

Dataset Generation and Training Scripts for CAV 2021.

## Installation and Required Packages
<hr /> 

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

Install unzip.
```
sudo apt-get install unzip
```

## Datasets
<hr /> 

The dataset that used for our experiments is the [M2NIST Dataset](https://www.kaggle.com/farhanhubble/multimnistm2nist). The dataset was created in order to teach the basics of semantic segmentation with convolutional neural networks without requiring the use of complex architechtures that take long to train. More details can be found at the above link. This dataset contains 5000 images with a shape of (64,84,1). Each segmentation mask contains 11 classes (Digits 0-9 and a background label), shape:(64,84,11). Segmented Examples are displayed below.

![Network Predictions](./matlab/readme_images/net75iou_predicted.png "Network Predictions")

For most segmentation tasks, 5000 images is not sufficient to obtain reasonable performance. Thus we created an additional 50000 images used to train the network. The original M2NIST dataset is only used for testing. If you wish to generate more images, the script we used to generate images can be found [here](dataset/generate_M2NIST.py).

### Lower Dimensional M2NIST

We also created a lower dimensional variant of the M2Nist. The images in this dataset have a shape of (28,56,1) and may contain either one or two digits. We generated 50,000 training images and 10,000 testing images. The script used to generate this dataset can be found [here](dataset/generate2856.py).

### MNIST MASKS

Finally, we also created a segmentation dataset using MNIST. Each image in this set has a shape of (28,28,1). Segmented Examples are shown below. The script used to generate this dataset can be found [here](dataset/generate_MNIST_masks.py).

![Network Predictions](./matlab/readme_images/mnist.png "Network Predictions")

## Setup 
<hr /> 
The experiments located in this respository were conducted using a machine with the following specifications:
```
OS: Ubuntu 16.04.6 LTS (Xenial Xerus), 4.15.0-45-generic x86_64
GPU(s): 1 GeForce GTX 1080
```
To reproduce the results:

### Generate the Datasets 

To generate all three datasets run the following:
```
chmod u+x setup.sh && ./setup.sh
```

## Training Models and Evaluating the Models
<hr /> 

The training scripts used to train the segmentation models are located in the [matlab](matlab) directory. 
Execute each of them to train segmentation models on the datasets generated above. We also provide a set of pre-trained networks for each of the aforementioned datasets.


# CAV2021 Networks 

The networks shown in Table 1 were trained using the following scripts. The model files will be stored in the [models](matlab/models/) directory. Each file is labled with the corresponding weighted iou.


- N1: [matlab/mnist.m](matlab/mnist.m) | weights: [net_mnist_3_relu.mat](matlab/models/mnist/net_mnist_3_relu.mat) 
- N2: [matlab/mnist_mp.m](matlab/mnist_mp.m) | weights: [net_mnist_3_relu_maxpool.mat](matlab/models/mnist/net_mnist_3_relu_maxpool.mat)
- N3: [matlab/mnist_dilated.m](matlab/mnist_dilated.m) | weights: [net_mnist_dilated_83iou.mat](matlab/models/mnist/net_mnist_dilated_83iou.mat)
- N4: [matlab/m2nist_dilated_ap.m](matlab/m2nist_dilated_ap.m)  | weights: [m2nist_62iou_dilatedcnn_avgpool.mat](matlab/models/m2nist/m2nist_62iou_dilatedcnn_avgpool.mat)
- N5: [matlab/m2nist_tranposed_training.m](matlab/m2nist_tranposed_training.m) | weights: [net75iou_avgpool.mat](matlab/models/m2nist/net75iou_avgpool.mat)
- N6: [matlab/m2nist_dilated.m](matlab/m2nist_dilated.m) | weights: [m2nist_dilated_72iou.mat](matlab/models/m2nist/m2nist_dilated_72iou.mat)


