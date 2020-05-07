%clear;
%close all;
%clc;

% load images
imds = imageDatastore('images');

% class names

classNames =["zero","one","two","three","four","five","six","seven","eight","nine","ten"];

pixelLabelID = [0,1,2,3,4,5,6,7,8,9,10];

pxds = pixelLabelDatastore('masks',classNames,pixelLabelID);

% count the pixels 

tbl = countEachLabel(pxds);

% fix class weighting imbalance

numberPixels = sum(tbl.PixelCount);
frequency = tbl.PixelCount / numberPixels;
classWeights = 1./ frequency;

% Visualize by pixel counts 

bar(1:numel(classNames),frequency);
xticks(1:numel(classNames));
xticklabels(tbl.Name)
xtickangle(45);
ylabel('Frequency');


% create the imageDataStore
plds= pixelLabelImageDatastore(imds,pxds);

% shuffle the dataset
plds = shuffle(plds);

% load the test set 
test_imds = imageDatastore('../matlab/images');
test_pxds = pixelLabelDatastore('../matlab/masks',classNames,pixelLabelID);
test_plds= pixelLabelImageDatastore(test_imds,test_pxds);
 
% Define Segmentation Network
numClasses = 11;
numFilters = 128;
filterSize = 3;
imageSize = [64,84,1];
layers = [
    imageInputLayer(imageSize)
    convolution2dLayer(filterSize,numFilters,'Padding',1)
    reluLayer()
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(filterSize,numFilters,'Padding',1)
    reluLayer()
    transposedConv2dLayer(4,numFilters,'Stride',2,'Cropping',1);
    convolution2dLayer(1,numClasses);
    softmaxLayer()
    pixelClassificationLayer('Name','labels','Classes',tbl.Name,'ClassWeights',classWeights)
    ];


% define optimizer
opts = trainingOptions('sgdm', ...
    'InitialLearnRate',1e-3, ...
    'MaxEpochs',100,...
    'ExecutionEnvironment','parallel',...
    'MiniBatchSize',64);

% train the network 
net = trainNetwork(plds,layers,opts);


% make predictions 
pxdsPred = semanticseg(test_plds,net,'MiniBatchSize', 64, 'WriteLocation','preds');

metrics = evaluateSemanticSegmentation(pxdsPred,test_plds);

save net
