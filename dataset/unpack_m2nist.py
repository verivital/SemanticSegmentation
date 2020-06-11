# convert .npy file to .mat
import numpy as np 
import cv2
import os
import sys
from pathlib import Path

# Whether or not you want to display image:
DISPLAY_IMAGE = False


images = np.load(os.path.join(str(Path(os.path.abspath(__file__)).parent),'combined.npy'))
segmentation_masks= np.load(os.path.join(str(Path(os.path.abspath(__file__)).parent),'segmented.npy'))

print("[INFO] Unpacking M2NIST")
print("{INFO] Mask Classes: ",np.unique(segmentation_masks.argmax(axis=-1)))

print(segmentation_masks.shape)

# This will load one of the images 
# Get the segmentation mask and then visualize the results
if DISPLAY_IMAGE:
    # get an image
    image= images[0]
    mask = segmentation_masks[0].astype(float)
    segmentation_mask=mask[0:mask.shape[0],0:mask.shape[1],10] = np.finfo(float).eps*10
    segmentation_mask = segmentation_mask.argmax(axis=-1)


    # calculate the distinct classes
    classes= np.unique(segmentation_mask)
    # create a new RGB image so we can visualize it
    output_image=np.zeros((image.shape[0],image.shape[1],3))

    # generate random rgb values
    colors = []
    for i in classes:
	    colors.append(np.random.rand(3,))

    # map each label to a randomly chosen color
    for j in range(len(classes)):
	    indices= np.where(segmentation_mask==classes[j])
	    output_image[indices]=colors[j]

    cv2.imshow("image",image/255.0)
    cv2.imshow("segmentation",output_image)
    cv2.waitKey(0)

# save image 
img_str = 'img_{}.png'
for (i,img) in enumerate(images):
    im_path=os.path.sep.join([str(Path(os.path.abspath(__file__)).parent),'m2nist/test_images',img_str.format(i+1)])
    mask_path=os.path.sep.join([str(Path(os.path.abspath(__file__)).parent),'m2nist/test_masks',img_str.format(i+1)])
    cv2.imwrite(im_path,img)

    mask = segmentation_masks[i].astype("float")
    mask[0:mask.shape[0],0:mask.shape[1],10] = np.finfo(float).eps*10
    mask = mask.argmax(axis=-1)
    
    # convert back to unit8
    masj=mask.astype('uint8')
    
    cv2.imwrite(mask_path, mask)

print("Finished Unpacking M2NIST...")