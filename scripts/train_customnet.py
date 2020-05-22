
# import libraries defined in this project
from nn.customNet import CustomNet
from utils.visualize import visualize_mask
from dataset_writer.hdf5datasetgenerator import HDF5DatasetGenerator
from callbacks.trainingmonitor import TrainingMonitor
from callbacks.epochcheckpoint import EpochCheckpoint


import cv2
import numpy as np 
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import MeanIoU
from dataset_writer.hdf5datasetgenerator import HDF5DatasetGenerator
from callbacks.trainingmonitor import TrainingMonitor
from callbacks.epochcheckpoint import EpochCheckpoint
import tensorflow.keras.backend as K 
import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from os import path 


# construct the argument parser and parse the arguments 
ap = argparse.ArgumentParser()
ap.add_argument("-c","--checkpoints",required=True,help="path to the output checkpoint directory")
ap.add_argument('-m','--model',type=str,help="path to *specific* model checkpoint to load")
ap.add_argument('-s','--start-epoch',type=int,default=0,help="epoch to restarting training at")
ap.add_argument('-tr','--training_record',type=str,default='records/train.hdf5',help="path to training hdf5 dataset")
ap.add_argument('-vr','--val_record',type=str,default='records/train.hdf5',help="path to validataion hdf5 dataset")
args = vars(ap.parse_args())


# Initialize the dataset generators 

trainGen = HDF5DatasetGenerator(args['training_record'],64,preprocessors=None)
valGen   = HDF5DatasetGenerator(args['val_record'],64,preprocessors=None)

# Constants
VISUALIZE_FIRST = False
NUM_EPOCHS = 100
BATCH_SIZE = 64

# load and normalize the data
if VISUALIZE_FIRST:
   ims,masks = next(trainGen.generator()) 
   cv2.imshow("original",ims[0])
   visualize_mask(masks[0])
   cv2.waitKey(0)
   trainGen.close()
   trainGen = HDF5DatasetGenerator(args['training_record'],64,preprocessors=None)



# if there is no specific model checkpoint supplied, then intialize the network and compile the model 
if args['model'] is None:
    print("[INFO] compiling model...")
    model = CustomNet.build(64,84,1,11)
    opt=SGD(lr=0.05,decay=0.01/NUM_EPOCHS,momentum=0.9,nesterov=True)
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=[MeanIoU(num_classes=11)])

# otherwise, load the checkpoint from disk 
else: 
    print("[INFO] loading {}...".format(args['model']))
    model = load_model(args['model'])

    # update the learning rate
    print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))

    K.set_value(model.optimizer.lr,1e-5)

    print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.lr)))


# construct the set of callbacks
# -- EpochCheckpoint will save a checkpoint file every 5 epochs
# -- TrainingCheckpoint will save the loss and accuracy
    # in a log file and plot for every epoch of training

callbacks = [
    EpochCheckpoint(args['checkpoints'],every=5,startAt=args['start_epoch']),
    TrainingMonitor(path.sep.join(['logs','custom_net.png']),
                    jsonPath=path.sep.join(['logs','custom_net.json']),startAt=args['start_epoch'])
]


# train the network 
model.fit_generator(
    trainGen.generator(),
    steps_per_epoch = trainGen.numImages // BATCH_SIZE, 
    validation_data= valGen.generator(),
    validation_steps= valGen.numImages // BATCH_SIZE,
    epochs = 10,
    max_queue_size = 64 *2,
    callbacks = callbacks,
    verbose =1
)

# close the databases
trainGen.close()
valGen.close()
