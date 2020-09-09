# import the necessary packages
from conv.unet import Unet
from utils.load_data import ImageUtils


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#import tensorflow libraries
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K  
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU

import matplotlib.pyplot as plt
import numpy as np
import argparse


# sometimes GPU systems are annoying 
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import Session
config = ConfigProto()
config.gpu_options.allow_growth = True
sess = Session(config=config)
sess.as_default()

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,help="path to images directory")
ap.add_argument("-m", "--masks", required=True,help="path to masks directory")
ap.add_argument("-o", "--output", required=True,help="directory where we will output the model")
args = vars(ap.parse_args())


ims,msks = ImageUtils.load_from_directory(args['images'],args['masks'],128,128,verbose=1,grayscale=1)
ims = ims/255.0
msks = msks/255


X_train, X_valid, y_train, y_valid = train_test_split(ims, msks, test_size=0.1, random_state=42)

model = Unet.build(128,128,1,1)

model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=[MeanIoU(num_classes=2)])


callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.0001, verbose=1),
    ModelCheckpoint(args['output'], verbose=1, save_best_only=True, save_weights_only=False)
]


results = model.fit(X_train, y_train, batch_size=32, epochs=50, callbacks=callbacks,validation_data=(X_valid, y_valid))

