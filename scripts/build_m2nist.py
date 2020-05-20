# import sys so we can use packages outside of this folder in
# either python 2 or python 3,
import sys
import os
from pathlib import Path 
#insert parent directory into the path
sys.path.insert(0,str(Path(os.path.abspath(__file__)).parent.parent))

from sklearn.model_selection import train_test_split
from dataset_writer.hdf5datasetwriter import HDF5DatasetWriter
from tensorflow.python.keras.utils import np_utils
from imutils import paths 
import numpy as np 
import progressbar
import json 
import cv2 
import os 

# Constants
IMAGES = '../dataset/images'
MASKS =  '../dataset/masks'
# grab the paths to the training images
im_paths=sorted(list(paths.list_images(IMAGES)))

# grab the masks
mask_paths=sorted(list(paths.list_images(MASKS)))


# Our intention is to create HDF5 files so we need to split the data beforehand
# previously we set the number of testing images to be 10% or 50 images per class
split = train_test_split(im_paths,mask_paths,test_size=0.1, random_state=42)

(trainPaths, testPaths, trainLabels, testLabels) = split 

# construct a list pairing the training validation, and testing 
# image paths along with their corresponding labels and output HDF5 
# files

datasets = [
    ("train",trainPaths,trainLabels,'records/train.hdf5'),
    ("val",testPaths,testLabels,'records/val.hdf5')
]


# loop over the dataset tuples

for (dType,paths,labels,outputPath) in datasets:
    
    # create the HDF5 writer
    print("[INFO] building {}...".format(outputPath))

    writer = HDF5DatasetWriter((len(paths),64,84,1),(len(paths),64,84,11),outputPath)

    # initialize the progressbar

    widgets = ["Building Dataset: ",progressbar.Percentage()," ",
              progressbar.Bar()," ",progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths),widgets=widgets).start()

    # loop over the image paths

    for (i,(path,label)) in enumerate(zip(paths,labels)):
        # load the image from disk
        image = cv2.imread(path,cv2.IMREAD_GRAYSCALE).astype('float32').reshape((64,84,1))
        # if we are building the training dataset, then compute the mean of each channel
        # then update the respective lists 
        label = cv2.imread(label,cv2.IMREAD_GRAYSCALE)
        label= np_utils.to_categorical(label,num_classes=11,dtype="float32")

        # add the image and label to the HDF5 dataset 
        writer.add([image],[label])
        pbar.update(i)

    # close the HDF5 writer
    pbar.finish()
    writer.close()




