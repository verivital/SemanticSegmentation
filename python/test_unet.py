# import the necessary packages
from utils.load_data import ImageUtils
from tensorflow.keras.models import load_model
import cv2 
import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn.metrics import classification_report

# sometimes GPU systems are annoying 
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import Session
config = ConfigProto()
config.gpu_options.allow_growth = True
sess = Session(config=config)
sess.as_default()

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,help="path to images directory")
ap.add_argument("-m", "--masks", required=True,help="path to masks directory")
ap.add_argument("-n", "--network", required=True,help="path to model file")
args = vars(ap.parse_args())


ims,msks = ImageUtils.load_from_directory(args['images'],args['masks'],128,128,verbose=1,grayscale=1)
ims = ims/255.0
msks = msks/255.0



model = load_model(args["network"])
preds = model.predict(ims)
#print(classification_report(msks, preds))


select= np.random.randint(0,ims.shape[0],100)

display_ims = ims[select]
display_preds = preds[select] 
display_masks = msks[select] 

for i in range(100):
    cv2.imshow("im",display_ims[i])
    cv2.imshow("pred",display_preds[i])
    cv2.imshow("mask",display_masks[i])
    cv2.waitKey(0)

