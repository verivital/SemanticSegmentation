% load images
imds = imageDatastore('images');

% class names

classNames =["zero","one","two","three","four","five","six","seven","eight","nine","ten"];

pixelLabelID = [0,1,2,3,4,5,6,7,8,9,10];

pxds = pixelLabelDatastore('masks',classNames,pixelLabelID);

% count the pixels 

tbl = countEachLabel(pxds);

% fix class weighting imbalance
% this one uses the median of the frequency class weights
imageFreq = tbl.PixelCount./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq;
% Visualize by pixel counts 

bar(1:numel(classNames),imageFreq);
xticks(1:numel(classNames));
xticklabels(tbl.Name)
xtickangle(45);
ylabel('Frequency');



% Use Data Augmentation during training, this helps provide more examples,
% because it helps improve the accuracy of the network. This one was used
% in the matlab example online, let's see how it works

augmenter = imageDataAugmenter('RandXReflection', true, ...
                               'RandXTranslation',[-5,5], ...
                               'RandYTranslation',[-5,5]);
                                


% create the imageDataStore
plds= pixelLabelImageDatastore(imds,pxds,'DataAugmentation',augmenter);

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
    'InitialLearnRate',1e-2, ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',33,...
    'LearnRateDropFactor',0.3,...
    'MaxEpochs',100,...
    'Momentum', 0.9,...
    'ExecutionEnvironment','parallel',...
    'MiniBatchSize',64, ...
    'Plots','training-progress',...
    'ValidationPatience',10);

% train the network 
net = trainNetwork(plds,layers,opts);


% make predictions 
pxdsPred = semanticseg(test_plds,net,'MiniBatchSize', 64, 'WriteLocation','preds');

metrics = evaluateSemanticSegmentation(pxdsPred,test_plds);

save aug_net



