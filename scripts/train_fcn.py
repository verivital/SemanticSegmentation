# import the necessary packages
from nn.customNet import CustomNet
from utils.visualize import visualize_mask
from imutils import paths
import cv2
import numpy as np 
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# Constants
IMAGES = '../dataset/images'
MASKS =  '../dataset/masks'
VISUALIZE_FIRST = False
NUM_EPOCHS = 100
BATCH_SIZE = 1

im_paths=sorted(list(paths.list_images(IMAGES)))
mask_paths=sorted(list(paths.list_images(MASKS)))


# load and normalize the data
images = []
masks = []
for i in range(len(im_paths)):
    img = cv2.imread(im_paths[i],cv2.IMREAD_GRAYSCALE).astype('float32')/255.0
    img = img.reshape((64,84,1))
    mask = cv2.imread(mask_paths[i],cv2.IMREAD_GRAYSCALE)
    mask = np_utils.to_categorical(mask,num_classes=11,dtype="float32") 
    images.append(img)
    masks.append(mask)
    if((i+1) % 500 == 0):
        print("[INFO] Processed {} Images...".format(i+1))

images = np.asarray(images)
masks = np.asarray(masks)
(trainX, testX, trainY, testY) = train_test_split(images,masks, test_size=0.10, random_state=42)

if VISUALIZE_FIRST:
   cv2.imshow("original",images[0])
   visualize_mask(masks[0])
   cv2.waitKey(0)

# build the model 
model = CustomNet.build(64,84,1,11)
print(model.summary())

opt=SGD(lr=0.05,decay=0.01/NUM_EPOCHS,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

#save the best performing models
fname='models/customnet1.hdf5'
checkpoint = ModelCheckpoint(fname, monitor="val_loss", mode="min",save_best_only=True,save_weights_only=False, verbose=1)

#Let us now instantiate th callbacks
callbacks=[checkpoint]
H = model.fit(trainX, trainY, validation_data=(testX, testY),batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=1,callbacks=callbacks)

# plot the training + testing loss and accuracy
plt.style.use("ggplot")
plt.figure()

for i in H.history.keys():
    if i=="loss":
        continue
    plt.plot(np.arange(0, NUM_EPOCHS), H.history[i], label=i)

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()