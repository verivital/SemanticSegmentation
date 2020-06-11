#!/bin/bash

# create the directories for the m2nist, mnist2856, mnist
mkdir -p dataset/m2nist/images
mkdir -p dataset/m2nist/masks
mkdir -p dataset/m2nist/test_images
mkdir -p dataset/m2nist/test_masks

mkdir -p dataset/m2nist2856/images
mkdir -p dataset/m2nist2856/masks
mkdir -p dataset/m2nist2856/test_images
mkdir -p dataset/m2nist2856/test_masks

mkdir -p dataset/mnist/images
mkdir -p dataset/mnist/masks
mkdir -p dataset/mnist/test_images
mkdir -p dataset/mnist/test_masks

mkdir -p dataset/preds
mkdir -p dataset/mnist_preds
mkdir -p dataset/mnist2856_preds

# generate the datasets
python dataset/unpack_m2nist.py && \
python dataset/generate_M2NIST.py -n 50000 -d 3 -i dataset/m2nist/images -m dataset/m2nist/masks && \
python dataset/generate2856.py -n 50000 -i dataset/m2nist2856/images -m dataset/m2nist2856/masks && \
python dataset/generate2856.py -n 10000 -i dataset/m2nist2856/test_images -s 15 -m dataset/m2nist2856/test_masks && \
python dataset/generate_MNIST_masks.py -i dataset/mnist/images -m dataset/mnist/masks && \
python dataset/generate_MNIST_masks.py -i dataset/mnist/test_images -m dataset/mnist/test_masks -t 1 





