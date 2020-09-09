#import the necessary packages
from imutils import paths
import os
import cv2
import imutils
import numpy as np

"""Class for loading images from a directory, probably more later.
    Assumes data is structured in the following format: data/{classification}/{time-stamp}~{command}.jpg
"""

class ImageUtils:
    @staticmethod
    def load_from_directory(image_directory,mask_directory,height,width,verbose=0,grayscale = 0):
        #count to show the user progress
        count=0
        images =[]
        masks = []
        # loop over the input images
        for imagePath in sorted(list(paths.list_images(image_directory))):
            
            split_path=os.path.split(imagePath)
            maskPath = os.path.join(mask_directory,split_path[-1])
            
            if(grayscale):
                img = cv2.imread(imagePath,cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(imagePath,cv2.IMREAD_COLOR)
            msk = cv2.imread(maskPath ,cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img,(width,height))
            msk = cv2.resize(msk,(width,height))
            #split the path
            
            images.append(img)
            masks.append(msk)
            count+=1
        
            if (count+1) % 500 == 0:
                print("INFO loaded: ",count+1,"images")

        images = np.asarray(images)
        masks = np.asarray(masks)
        if grayscale:
            images = images.reshape((images.shape[0],images.shape[1],images.shape[2],1))
        masks = masks.reshape((masks.shape[0],masks.shape[1],masks.shape[2],1))

        return images,masks
        

# FOR TESTING PURPOSES
if __name__== "__main__":
    imsd= '/home/musaup/Documents/Research/Segmentation/python/data/competition_data/train/images/'
    masks= '/home/musaup/Documents/Research/Segmentation/python/data/competition_data/train/masks/'
    ims,msks = ImageUtils.load_from_directory(imsd,masks,128,128,verbose=1,grayscale=1)
    print(ims.shape,msks.shape)
    print(msks[1]/255)