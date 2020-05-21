# import the neccessary packages
from tensorflow.keras.callbacks import Callback
import os

class EpochCheckpoint(Callback):
    def __init__(self,outputPath,every=5,startAt=0):

        # call the parent constructor
        # The base contructor has two attributes 
        # params: Dict, Training parameters(e.g verbosity,batch_size,number of epochs...)
        # model: Instance of keras.models.Model. Reference to the model that is being trained
        super(Callback,self).__init__()

        # store the base output path for the model, the number of epochs that must 
        # pass before the model is serialized to disk and the current epoch value 

        self.outputPath=outputPath
        self.every=every
        self.initEpoch = startAt

    def on_epoch_end(self,epoch,logs={}):
        # check to see if the model should be serialized to disk
        if(self.initEpoch+1) % self.every == 0:
            p=os.path.sep.join([self.outputPath,
            "epoch_{}.hdf5".format(self.initEpoch+1)])
            self.model.save(p,overwrite=True)
        # increment the internal epoch counter
        self.initEpoch +=1