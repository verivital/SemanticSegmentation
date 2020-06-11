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
from pathlib import Path
np.set_printoptions(threshold=sys.maxsize)

""" This code generates the M2NIST Dataset and was inspired by Farhan Ahmad
    (https://github.com/farhanhubble/udacity-connect/blob/master/segmented-generator.ipynb)
 """


class M2NIST:

    def __init__(self,img_dir,mask_dir,size=5000,digits_per_image=3,random_seed=1234):
        self.digits_per_image=digits_per_image
    
        self.size = size 
        self.img_dir = img_dir
        self.mask_dir = mask_dir

        # Load the MNIST Dataset
        (self.x_train, self.y_train), (_, _) = mnist.load_data()
        
        # Shuffle the dataset

        self.x_train,self.y_train = shuffle(self.x_train,self.y_train,random_state=random_seed)

        # Specify the canvas size
        self.canvas_size=(64,84)
        
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

        # save images and masks to the specified directories
        self.save_img_mask()

        # visualize if you want
        #self.visualize_images()

        
              

        # randomly select images from the MNIST dataset

    def generate_image_segmentation_pair(self):
        # Generate a random number between 2 and digits per image 
        # that specifies how many digits will go into the output image
        nb_digits = np.random.randint(low=2,high=self.digits_per_image+1)

        # Based on the number of images to be placed in the output image
        # randomly select three images from the training set, using a random
        # seed

        rand_indices = np.random.randint(0,len(self.x_train),nb_digits)

        src_images = self.x_train[rand_indices]
        src_labels =self.y_train[rand_indices]

        # create output image for the segmentation mask
        labels = np.zeros([self.canvas_size[0],self.canvas_size[1],11],dtype=self.x_train[0].dtype)

        # randomly generate locations in the image to place the digits
        boxes = self.generate_areas_to_copy()
        boxes = boxes[:nb_digits]

        # loop through the number of images to be placed in the output canvas
        for i in range(len(boxes)):

            x_off_start = boxes[i][0]
            y_off_start = boxes[i][1]
            x_off_end = boxes[i][2]
            y_off_end = boxes[i][3] 

            if x_off_end <= self.canvas_size[0] and y_off_end <= self.canvas_size[1]:
                src_img = src_images[i]
                src_digit = src_labels[i]
                labels[x_off_start:x_off_end,y_off_start:y_off_end,src_digit] = src_img

            

        image = np.max(labels, axis=2)
        mask = np.clip(labels,a_min=0,a_max=1)

        return image,mask





    def generate_areas_to_copy(self):
        
        # generate 100 boxes so that we can choose the ones with minimum ious
        x_starts = np.random.randint(0,self.canvas_size[0]-28,100)
        x_ends = x_starts+28

        y_starts = np.random.randint(0,self.canvas_size[1]-28,100)
        y_ends = y_starts + 28

        boxes = np.zeros((100,4))
        boxes[:,0] = x_starts
        boxes[:,1] = y_starts
        boxes[:,2] = x_ends
        boxes[:,3] = y_ends
        boxes = self.non_max_suppression_fast(boxes,0.15)
        return shuffle(boxes)


    def save_img_mask(self):
        print("[INFO] Generating M2NIST Images...")
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
        print("[INFO] Finished Generating M2NIST Images...")

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

        cv2.namedWindow('image',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 600,600)
        cv2.imshow("image",output_image)
        cv2.waitKey(0)



    # Malisiewicz et al. Non_max Suppression
    def non_max_suppression_fast(self,boxes, overlapThresh):
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []
        # if the bounding boxes integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")
        # initialize the list of picked indexes	
        pick = []
        # grab the coordinates of the bounding boxes
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]
            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last],
                np.where(overlap > overlapThresh)[0])))
        # return only the bounding boxes that were picked using the
        # integer data type
        return boxes[pick].astype("int")

        




if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num_images", required=False,type=int,help="number of training images to generate")
    ap.add_argument("-d", "--digits_per_image", required=False,type=int,help="how many images to put into the output image")
    ap.add_argument('-s','--seed',required=False,type=int,default=15,help="random seed used to generate dataset")
    ap.add_argument("-i,","--img_dir",required=True,help="path to directory to store image files")
    ap.add_argument("-m","--mask_dir",required=True,help="path to directory to store mask files")
    args = vars(ap.parse_args())

    # Instantiate M2NIST Objects
    if(args['digits_per_image'] and args['num_images']):
        m2nist = M2NIST(args['img_dir'], args['mask_dir'],size=args['num_images'],
                        digits_per_image=args['digits_per_image'],random_seed=args['seed'])
    elif(args['digits_per_image']):
        m2nist = M2NIST(args['img_dir'], args['mask_dir'],digits_per_image=args['digits_per_image'],random_seed=args['seed'])
    elif (args['num_images']):
        m2nist = M2NIST(args['img_dir'], args['mask_dir'],size=args['num_images'],random_seed=args['seed'])
    else:
        mnist=M2NIST(args['img_dir'], args['mask_dir'],random_seed=args['seed'])

        