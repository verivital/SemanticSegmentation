import numpy as np
import cv2
from visualize import visualize_mask_img
import argparse
import os


ap = argparse.ArgumentParser()
ap.add_argument("-i,","--img_dir",required=True,help="path to directory to store image files")
ap.add_argument("-m","--mask_dir",required=True,help="path to directory to store mask files")
ap.add_argument("-n","--num_images",required=False,type=int,default=10,help="number of images to visualize")
args = vars(ap.parse_args())


for i in range(1,args['num_images']+1):
    # create a string name for the files
    img_str = os.path.join(args['img_dir'],"image_{}.png").format(i)
    mask_str = os.path.join(args['mask_dir'],"image_{}.png").format(i)

    print(img_str,mask_str)
    img = cv2.imread(img_str,cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_str,cv2.IMREAD_GRAYSCALE)

    visualize_mask_img(mask,verbose=1)
    cv2.imshow("Original Input Image",img)
    cv2.waitKey(0)