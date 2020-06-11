# Semantic Segmentation in MATLAB 

## MATLAB Version
These experiments were run using MATLAB 2020a. 

Toolboxes utilized:
- Deep Learning Toolbox
- Computer Vision Toolbox
- Parallel Computing Toolbox

# Train the Network

Each matlab scripts, will train a segmentation model on the dataset corresponding to the filename. Within each file we define the network architechture as well as specify the hyperparameters.

The [models](models) directory contains the models that we obtained by executing the above scripts. To evaluate these models simply load them into matlab and execute any of the evaluate_network scripts.

# Evaluate the network 

Once the network is trained you can evaluate it using the script [evaluate_network.m](evaluate_network.m) 

Once the predictions have been made you can visualize the segmentation predictions and obtain a plot such as the following:

### M2NIST Predictions 
![Network Predictions](./readme_images/net75iou_predicted.png "Network Predictions")

The above plot was produced by the network file [net_75iou.mat](./models/net_75iou.mat).

### MNIST Predictions 
![Network Predictions](./readme_images/mnist.png "Network Predictions")

The above plot was produced by the network file [net_mnist.mat](./models/net_mnist.mat).



