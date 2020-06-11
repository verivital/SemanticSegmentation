# import the neccessary packages 
import numpy as np
import cv2

def visualize_mask(mask,verbose=0):

    """
    Visualizes mask of shape (H,W,C) in openCV, where H = height, W = width, C = number of classes
    """

    segmentation_mask = mask.argmax(axis=-1)
    # bias the last entry in the mask
    classes= np.unique(segmentation_mask)

    if verbose:
        print(classes)
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

    cv2.namedWindow('mask',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('mask', 600,600)
    cv2.imshow("mask",output_image)


def visualize_mask_img(segmentation_mask,verbose=0):

    """
    Visualizes mask of shape (H,W,C) in openCV, where H = height, W = width, C = number of classes
    """

    # bias the last entry in the mask
    classes= np.unique(segmentation_mask)

    if verbose:
        print(classes)
    # create a new RGB image so we can visualize it
    output_image=np.zeros((segmentation_mask.shape[0],segmentation_mask.shape[1],3))

    # generate random rgb values
    colors = []
    for i in classes:
	    colors.append(np.random.rand(3,))

    # map each label to a randomly chosen color
    for j in range(len(classes)):
	    indices= np.where(segmentation_mask==classes[j])
	    output_image[indices]=colors[j]

    cv2.namedWindow('mask',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('mask', 600,600)
    cv2.imshow("mask",output_image)
