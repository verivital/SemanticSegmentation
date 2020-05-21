#import the neccessary packages 
from tensorflow.python.keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np 
import json 
import os

#This class inherits from BaseLogger which a callback that accumulates 
#averages of metrics. It is automatically applied to all models

class TrainingMonitor(BaseLogger):
    def __init__(self,figPath,jsonPath=None,startAt=0):
        #store the output path for the figure, the path to
        # the JSON serialized file, and the starting epoch

        #This is how you call a super constructor in python (new things e'ry day)
        #the startAt is the epoch where training is resumed when using ctrl + c training.
        super(TrainingMonitor,self).__init__() 
        self.figPath=figPath
        self.jsonPath=jsonPath
        self.startAt=startAt

    #callback that executes when the training process starts
    def on_train_begin(self,logs={}):
        #initialize the history dictionary
        self.H={}

        #if the JSON history path exists, load the training history 
        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.H=json.loads(open(self.jsonPath).read())

                #check to see if a starting epoch was supplied 
                if self.startAt >0:
                    #loop over the entries in the history log and trim any entries that are past the starting epoch
                    for k in self.H.keys():  #this includes things like accuracy loss, validation so remove things past the starting epoch
                        self.H[k]=self.H[k][:self.startAt]

    #The on_epoch_end method is automatically supplied to parameters from Keras.
    #The first is an integer representing the epoch number. The second is a dictionary, logs, which contains the
    #training and validation loss + accuracy for the current epoch
    def on_epoch_end(self,epoch,logs={}):
        #loop over the logs and update the loss, accuracy, etc.
        #for the entire training process
        for (k,v) in logs.items(): #.items() expands the dictionary into a list of tuples
            #so what happens is we get the list from this class, append the new value from the logs, and store it again
            l=self.H.get(k,[]) #this takes to params it retunrs the value if the key is in the dictionary, second param is what is returned if the key doesn't exist
            #you have to convert the keras floats to python floats
            l.append(float(v))
            self.H[k]=l
    
        #check to see if the training history should be serialized to file
        if self.jsonPath is not None:
            f=open(self.jsonPath,'w')
            f.write(json.dumps(self.H))
            f.close()

        #Ensure that at least two epochs have passed before plotting 
        #(epoch starts at zero)

        if(len(self.H['loss'])>1):
            # plot the training loss and accuracy
            N = np.arange(0, len(self.H["loss"]))
            plt.style.use("ggplot")
            plt.figure()
            for key in self.H.keys():
                plt.plot(N, self.H[key], label=key)
            plt.title("Training Loss and Accuracy [Epoch {}]".format(len(self.H["loss"])))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()

            #plt.show()
            #you can save the figure
            plt.savefig(self.figPath)
            plt.close()






  