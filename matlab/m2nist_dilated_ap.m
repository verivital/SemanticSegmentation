% load images
imds = imageDatastore('../dataset/m2nist/images');

% class names

classNames =["zero","one","two","three","four","five","six","seven","eight","nine","ten"];

pixelLabelID = [0,1,2,3,4,5,6,7,8,9,10];

pxds = pixelLabelDatastore('../dataset/m2nist/masks',classNames,pixelLabelID);

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
test_imds = imageDatastore('../dataset/m2nist/test_images/');
test_pxds = pixelLabelDatastore('../dataset/m2nist/test_masks/',classNames,pixelLabelID);
test_plds= pixelLabelImageDatastore(test_imds,test_pxds);



% Define Segmentation Network
numClasses = 11;
numFilters = 128;
imageSize = [64,84,1];
layers = [
    imageInputLayer(imageSize,'Name','input')
    
    % block 1
    convolution2dLayer(3,128,'Padding','same','Name','conv1_1')
    convolution2dLayer(3,128,'Padding','same','Name','conv1_2')
    reluLayer('Name','relu1_2')
    averagePooling2dLayer(2,'Stride',1,'Name','pool_1','Padding','same')
    
    % block 2
    convolution2dLayer(3,256,'DilationFactor',2,'Padding','same')
    convolution2dLayer(3,256,'DilationFactor',2,'Padding','same')
    reluLayer()
    averagePooling2dLayer(2,'Stride',1,'Padding','same')
    

    % block 3
    convolution2dLayer(3,512,'Padding','same')
    convolution2dLayer(3,512,'DilationFactor',2,'Padding','same')
    reluLayer()
    averagePooling2dLayer(2,'Stride',1,'Padding','same')
    
    % class layer
    convolution2dLayer(1,numClasses);
    softmaxLayer()
    pixelClassificationLayer('Name','labels','Classes',tbl.Name,'ClassWeights',classWeights)
 ];

analyzeNetwork(layers)



% define optimizer
opts = trainingOptions('sgdm', ...
    'InitialLearnRate',2e-3, ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',3,...
    'LearnRateDropFactor',0.5,...
    'MaxEpochs',9,...
    'Momentum', 0.9,...
    'ExecutionEnvironment','gpu',...
    'MiniBatchSize',32, ...
    'Plots','training-progress',...
    'ValidationPatience',10);

% train the network 
net = trainNetwork(plds,layers,opts);


% make predictions 
pxdsPred = semanticseg(test_plds,net,'MiniBatchSize', 64, 'WriteLocation','../dataset/preds/');

metrics = evaluateSemanticSegmentation(pxdsPred,test_plds);

filename = strcat('models/m2nist/m2nist_dilated_ap',sprintf('%.2f',metrics.DataSetMetrics.WeightedIoU),'_iou.mat');
save(filename,'net');



