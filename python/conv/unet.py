# import the necessary packages
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.convolutional import AveragePooling2D
from tensorflow.python.keras.layers.convolutional import MaxPooling2D
from tensorflow.python.keras.layers.core import Activation 
from tensorflow.python.keras.layers.core import Dropout 
from tensorflow.python.keras.layers.convolutional import Conv2DTranspose
from tensorflow.python.keras.layers.core import Dense 
from tensorflow.python.keras.layers import Flatten 
from tensorflow.python.keras.layers import Input 
from tensorflow.python.keras.models  import Model 
from tensorflow.python.keras.layers import concatenate
from tensorflow.python.keras.regularizers import l2 
from tensorflow.python.keras import backend as K


"""
Unet implementation as described in: https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47
"""
class Unet:

    @staticmethod
    def conv2d_block(input_tensor,n_filters,kernel_size = 3, batchnorm=True):
        """function that adds 2 convolutional layers with the parameters passed to it"""
        x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size), padding = 'same')(input_tensor)
        x = Activation('relu')(x)
        if batchnorm:
            x = BatchNormalization()(x)

        x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size), padding = 'same')(x)
        x = Activation('relu')(x)
        if batchnorm:
            x = BatchNormalization()(x)
        return x


    @staticmethod
    def build(height,width,depth,classes):
        inputShape = (height,width,depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension 

        if K.image_data_format() == "channels_first":
            inputShape = (depth,height, width)
            chanDim = 1

        # Define the model input and first CONV module
        inputs = Input(shape=inputShape)
        
        # filter and dropout parameters
        n_filters = 16
        dropout = 0.1
        batchnorm_param = True
        kernel_size = 3

        # Encoder
        
        c1 = Unet.conv2d_block(inputs,n_filters,kernel_size, batchnorm_param)
        p1 = MaxPooling2D((2, 2))(c1)
        p1 = Dropout(dropout)(p1)

        c2 = Unet.conv2d_block(p1,n_filters*2,kernel_size, batchnorm_param)
        p2 = MaxPooling2D((2, 2))(c2)
        p2 = Dropout(dropout)(p2)

        c3 = Unet.conv2d_block(p2,n_filters*4,kernel_size, batchnorm_param)
        p3 = MaxPooling2D((2, 2))(c3)
        p3 = Dropout(dropout)(p3)

        c4 = Unet.conv2d_block(p3,n_filters*8,kernel_size, batchnorm_param)
        p4 = MaxPooling2D((2, 2))(c4)
        p4 = Dropout(dropout)(p4)

        c5 = Unet.conv2d_block(p4,n_filters*16,kernel_size, batchnorm_param)


        # Decoder
        u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
        u6 = concatenate([u6, c4])
        u6 = Dropout(dropout)(u6)
        c6 = Unet.conv2d_block(u6, n_filters * 8, kernel_size, batchnorm_param)

        u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
        u7 = concatenate([u7, c3])
        u7 = Dropout(dropout)(u7)
        c7 = Unet.conv2d_block(u7, n_filters * 4, kernel_size, batchnorm_param)

        u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
        u8 = concatenate([u8, c2])
        u8 = Dropout(dropout)(u8)
        c8 = Unet.conv2d_block(u8, n_filters * 2, kernel_size, batchnorm_param)

        u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
        u9 = concatenate([u9, c1])
        u9 = Dropout(dropout)(u9)
        c9 = Unet.conv2d_block(u9, n_filters * 1, kernel_size, batchnorm_param)

        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
        model = Model(inputs=[inputs], outputs=[outputs])

        return model
