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



class M2NIST_2856:
    def __init__(self,img_dir,mask_dir,size=5000,canvas_size=(28,56),digits_per_image=2,random_seed=1234):
            
            self.digits_per_image=digits_per_image
            self.size = size 
            self.img_dir = img_dir
            self.mask_dir = mask_dir


            # Load the MNIST Dataset
            (self.x_train, self.y_train), (_, _) = mnist.load_data()
            
            # Shuffle the dataset

            self.x_train,self.y_train = shuffle(self.x_train,self.y_train,random_state=random_seed)

            # Specify the canvas size
            self.canvas_size=canvas_size

            # Set the random seed
            np.random.seed(random_seed)

            # lists for images and segementation masks
            self.images=[]
            self.masks =[]

            # generate mask and image pairs
            for i in range(self.size):
                img,mask= self.generate_image_segmentation_pair()
                self.images.append(img)
                self.masks.append(mask)

            self.save_img_mask()
            # visualize images
            #self.visualize_images()

    def generate_image_segmentation_pair(self):
        # Generate a random number between 2 and digits per image 
        # that specifies how many digits will go into the output image
        nb_digits = np.random.randint(low=1,high=3)

        # Based on the number of images to be placed in the output image
        # randomly select three images from the training set, using a random
        # seed

        rand_indices = np.random.randint(0,len(self.x_train),nb_digits)

        src_images = self.x_train[rand_indices]
        src_labels =self.y_train[rand_indices]

        # create output image for the segmentation mask
        labels = np.zeros([self.canvas_size[0],self.canvas_size[1],11],dtype=self.x_train[0].dtype)

        # randomly generate locations in the image to place the digits
        boxes = [[0,0,28,28], [28,0,56,28]]


        # loop through the number of images to be placed in the output canvas
        for i in range(nb_digits):

            box = np.random.randint(low=0,high=len(boxes))

            x_off_start = boxes[box][0]
            y_off_start = boxes[box][1]
            x_off_end = boxes[box][2]
            y_off_end = boxes[box][3]

            boxes.pop(box)
           
        
            src_digit = src_labels[i]
            src_img = src_images[i]
            labels[y_off_start:y_off_end,x_off_start:x_off_end,src_digit] = src_img
        image = np.max(labels, axis=2)
        mask = np.clip(labels,a_min=0,a_max=1)

        return image,mask

    def save_img_mask(self):
        print("[INFO] Generating M2NIST 28x56 Images...")
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
        print("[INFO] Finished Generating M2NIST 2856 Images...")

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
    ap.add_argument("-n", "--num_images", required=False,type=int,help="number of training images to generate")
    ap.add_argument("-i,","--img_dir",required=True,help="path to directory to store image files")
    ap.add_argument("-m","--mask_dir",required=True,help="path to directory to store mask files")
    ap.add_argument('-s','--seed',required=False,type=int,default=15,help="random seed used to generate dataset")
    args = vars(ap.parse_args())

    mnist_2856= M2NIST_2856(args['img_dir'],args['mask_dir'],size=args['num_images'],random_seed=args['seed']) 
