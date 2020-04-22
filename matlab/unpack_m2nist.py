# convert .npy file to .mat
import numpy as np 
import cv2
import os

# Whether or not you want to display image:
DISPLAY_IMAGE = False

images = np.load('combined.npy')
segmentation_masks= np.load('segmented.npy')

print("{INFO] Mask Classes: ",np.unique(segmentation_masks.argmax(axis=-1)))

print(segmentation_masks.shape)

# This will load one of the images 
# Get the segmentation mask and then visualize the results
if DISPLAY_IMAGE:
    # get an image
    image= images[0]
    segmentation_mask = segmentation_masks[0].argmax(axis=-1)

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

# create directories
if not os.path.isdir('images'):
    os.mkdir('images')
if not os.path.isdir('masks'):
    os.mkdir('masks')
if not os.path.isdir('preds'):
    os.mkdir('preds')

# save image 
img_str = 'img_{}.png'
for (i,img) in enumerate(images):
    im_path=os.path.join('images',img_str.format(i+1))
    mask_path=os.path.join('masks',img_str.format(i+1))
    cv2.imwrite(im_path,img)
    cv2.imwrite(mask_path,segmentation_masks[i].argmax(axis=-1))