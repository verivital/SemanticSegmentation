# Semantic Segmentation in MATLAB ... 


## MATLAB Version
These experiments were run using MATLAB 2020a. If you have a previous version, I can't say if it will work or not.

Toolboxes utilized:

- Deep Learning Toolbox
- Computer Vision Toolbox
- Parallel Computing Toolbox

## Datasets

The dataset that I used for my experiments is the [M2NIST Dataset](https://www.kaggle.com/farhanhubble/multimnistm2nist). The dataset was created in order to teach the basics of semantic segmentation with convolutional neural networks without requiring the use of complex architechtures that take long to train. More details can be found at the above link.

## Setup and Installation

Download the numpy binary files combined.npy and segmented from [kaggle]([https://www.kaggle.com/farhanhubble/multimnistm2nist]).

To create the image directories I used python and OpenCV. If you don't have OpenCV install it by running the following:

```
$ pip install numpy
$ pip install opencv 
```

If you already have Numpy and OpenCV run the following:

```
$ python unpack_m2nist.py
```

This will create three directories: 
- masks contains the segmentations masks
- images contains the input images
- preds will contain the predictions once a network is trained.

# Train the Network

After completing the above setup. Running the script [encoder_decoder.m](encoder_decoder.m) will train a simple [encoder-decoder segmentation network](https://courses.cs.washington.edu/courses/cse576/17sp/notes/Sachin_Talk.pdf).

The script will train and save a network named net.mat to this directory. You can play with the hyperparameters in with the above file. 

# Evaluate the network 

Once the network is trained you can evaluate it using the script [evaluate_network.m](evaluate_network.m) 



