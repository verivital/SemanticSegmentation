# Semantic Segmentation in MATLAB 

## MATLAB Version
These experiments were run using MATLAB 2020a. 

Toolboxes utilized:
- Deep Learning Toolbox
- Computer Vision Toolbox
- Parallel Computing Toolbox

# Train the Network

After completing the above setup. Running the script [encoder_decoder.m](encoder_decoder.m) will train a simple [encoder-decoder segmentation network](https://courses.cs.washington.edu/courses/cse576/17sp/notes/Sachin_Talk.pdf).

The script will train and save a network named net.mat to this directory. You can play with the hyperparameters in with the above file. 

# Evaluate the network 

Once the network is trained you can evaluate it using the script [evaluate_network.m](evaluate_network.m) 

Once the predictions have been made you can visualize the segmentation predictions and obtain a plot such as the following:

![Network Predictions](./readme_images/net75iou_predicted.png "Network Predictions")

The above plot was produced by the network file [net_75iou.mat](./models/net_75iou.mat).


![Network Predictions](./readme_images/mnist.png "Network Predictions")

The above plot was produced by the network file [net_mnist.mat](./models/net_mnist.mat).



