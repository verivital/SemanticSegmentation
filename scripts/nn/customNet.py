# import the necessary packages
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.convolutional import MaxPooling2D
from tensorflow.python.keras.layers.convolutional import Conv2DTranspose
from tensorflow.python.keras.layers.core import Activation
from tensorflow.python.keras.layers.core import Flatten
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.keras.layers.core import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K


class CustomNet:
    @staticmethod
    def build(height, width, depth, classes, reg=0.0002):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        
        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        
        # Block #1: Two conv then relu maxpool 
        model.add(Conv2D(64, (3, 3), strides=(1, 1),
            input_shape=inputShape, 
            padding="same",kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same",kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Block 2: Two conv then relu maxpool
        model.add(Conv2D(128, (3, 3), strides=(1, 1),
            input_shape=inputShape, 
            padding="same",kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(Conv2D(128, (3, 3), strides=(1, 1), padding="same",kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Block 3: Two conv then relu
        model.add(Conv2D(256, (3, 3), strides=(1, 1),
            input_shape=inputShape, 
            padding="same",kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(Conv2D(256, (3, 3), strides=(1, 1), padding="same",kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim)) 

        # Block 4: Upsample, Upsample, UpSample

        model.add(Conv2DTranspose(256,(2,2),strides=(2,2)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim)) 

        model.add(Conv2DTranspose(512,(2,2),strides=(2,2)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(classes, (3, 3), strides=(1, 1), padding="same",kernel_regularizer=l2(reg)))
        model.add(Activation("softmax"))



        # return the constructed network architecture
        return model


if __name__=="__main__":
    model=CustomNet.build(64,84,3,11)
    print(model.summary())