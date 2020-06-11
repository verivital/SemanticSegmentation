# import the neccessary packages
import matplotlib.pyplot as plt 
import numpy as np 
from keras.datasets import mnist
from keras.utils import to_categorical
import argparse
from sklearn.utils import shuffle 
import cv2 
import os 
import sys
np.set_printoptions(threshold=sys.maxsize)


class M2NIST_MASK:
    def __init__(self,img_dir,mask_dir,train_or_test=False):
            
            self.img_dir = img_dir
            self.mask_dir = mask_dir

            self.train_or_test=train_or_test

            # Load the MNIST Dataset
            (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
            
            # Shuffle the dataset
            self.x_train,self.y_train = shuffle(self.x_train,self.y_train,random_state=1617)

            # lists for images and segementation masks
            self.images=[]
            self.masks =[]

         
            self.generate_image_segmentation_pair()
            

            self.save_img_mask()
            # visualize images
            #self.visualize_images()

    def generate_image_segmentation_pair(self):
        
        if not self.train_or_test:
            src_images = self.x_train
            src_labels =self.y_train
        else:
            src_images = self.x_test
            src_labels =self.y_test



        for i in range(len(src_images)):
            # create output image for the segmentation mask
            labels = np.zeros([src_images[0].shape[0],src_images[0].shape[1],11],dtype=self.x_train[0].dtype)
            src_digit = src_labels[i]
            src_img= src_images[i]

            labels[:,:,src_digit] = src_img
            image = np.max(labels, axis=2)
            mask = np.clip(labels,a_min=0,a_max=1)

            self.images.append(image)
            self.masks.append(mask)

    def save_img_mask(self):
        print("[INFO] Generating M2NIST Masks...")
        count =1
        # create a string name for the files
        img_str = os.path.join(self.img_dir,"image_{}.png")
        mask_str = os.path.join(self.mask_dir,"image_{}.png")

        for (img,mask) in zip(self.images,self.masks):
            # bias last entry of background
            mask = mask.astype(float)
            mask[0:mask.shape[0],0:mask.shape[1],10] = np.finfo(float).eps*10
            segmentation_mask = mask.argmax(axis=-1)
            # calculate the distinct classes
            classes= np.unique(segmentation_mask)
            
            # convert images back to unint8
            segmentation_mask=segmentation_mask.astype('uint8')
            img=img.astype('uint8')

            # save the images
            cv2.imwrite(img_str.format(count),img)
            cv2.imwrite(mask_str.format(count),segmentation_mask)
            count +=1
        
        print("[INFO] Finished Generating M2NIST Masks...")

    def visualize_images(self):
        for (img,mask) in zip(self.images,self.masks):
            cv2.namedWindow('original image',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('original image', 600,600)
            cv2.imshow("original image",img)
            self.visualize_mask(mask) 

    def visualize_mask(self,mask):

        # bias the last entry in the mask
        mask = mask.astype(float)
        mask[0:mask.shape[0],0:mask.shape[1],10] = np.finfo(float).eps*10
        segmentation_mask = mask.argmax(axis=-1)
        # calculate the distinct classes
        classes= np.unique(segmentation_mask)

        # create a new RGB image so we can visualize it
        output_image=np.zeros((mask.shape[0],mask.shape[1],3))

        # generate random rgb values
        colors = []
        for i in classes:
	        colors.append(np.random.rand(3,))

        # map each label to a randomly chosen color
        for j in range(len(classes)):
	        indices= np.where(segmentation_mask==classes[j])
	        output_image[indices]=colors[j]
        cv2.imshow("mask",output_image)
        cv2.waitKey(0)



if __name__=='__main__':


    ap = argparse.ArgumentParser()
    ap.add_argument("-i,","--img_dir",required=True,help="path to directory to store image files")
    ap.add_argument("-m","--mask_dir",required=True,help="path to directory to store mask files")
    ap.add_argument('-t','--train_or_test',type=bool,default=False,help='flag that selects train or test. True corresponds to test')
    args = vars(ap.parse_args())

    mnist= M2NIST_MASK(args['img_dir'],args['mask_dir'],train_or_test=args['train_or_test'])
