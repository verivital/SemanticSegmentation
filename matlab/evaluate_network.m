load('net.mat');


% load images
imds_e = imageDatastore('images');

% class names

classNames =["zero","one","two","three","four","five","six","seven","eight","nine","ten"];

pixelLabelID = [0,1,2,3,4,5,6,7,8,9,10];

pxds_e = pixelLabelDatastore('masks',classNames,pixelLabelID);

% create the imageDataStore
plds= pixelLabelImageDatastore(imds_e,pxds_e);

% make predictions 
pxdsPred = semanticseg(plds,net,'MiniBatchSize', 64, 'WriteLocation','preds');

[test_image,test_image_gt]=readByIndex(plds,1);
[pred_image,pred_image_gt]=readByIndex(plds,1);

I = cell2mat(test_image.inputImage);
B = uint8(test_image.pixelLabelImage{:});
C = labeloverlay(I,B);
imshow(C);
figure()
P = cell2mat(pred_image.inputImage);
PB = uint8(pred_image.pixelLabelImage{:});
PC = labeloverlay(P,PB);
imshow(PC);

metrics = evaluateSemanticSegmentation(pxdsPred,plds);