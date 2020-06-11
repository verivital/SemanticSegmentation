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
imageSize = [64,84,1];
layers = [
    imageInputLayer(imageSize,'Name','input')
    
    % block 1
    convolution2dLayer(3,128,'DilationFactor',2,'Padding','same','Name','conv1_1')
    %convolution2dLayer(3,128,'DilationFactor',4,'Padding','same','Name','conv1_2')
    convolution2dLayer(3,128,'DilationFactor',2,'Padding','same','Name','conv1_3')
    %convolution2dLayer(3,128,'DilationFactor',4,'Padding','same','Name','conv1_4')
    reluLayer('Name','relu1')
    batchNormalizationLayer('Name','BN1')
    
    % block 2
    convolution2dLayer(3,256,'DilationFactor',4,'Padding','same','Name','conv2_1')
    %convolution2dLayer(3,256,'DilationFactor',8,'Padding','same','Name','conv2_2')
    convolution2dLayer(3,256,'DilationFactor',4,'Padding','same','Name','conv2_3')
    %convolution2dLayer(3,256,'DilationFactor',8,'Padding','same','Name','conv2_4')
    reluLayer('Name','relu2')
    batchNormalizationLayer('Name','BN2')
    
    
    % block 3
    convolution2dLayer(3,512,'DilationFactor',8,'Padding','same','Name','conv3_1')
    convolution2dLayer(3,512,'DilationFactor',8,'Padding','same','Name','conv3_2')
    reluLayer('Name','relu3')
    batchNormalizationLayer('Name','BN3')
    
    % block 4
    convolution2dLayer(3,256,'DilationFactor',4,'Padding','same','Name','conv4_1')
    convolution2dLayer(3,256,'DilationFactor',4,'Padding','same','Name','conv4_2')
    reluLayer('Name','relu4')
    batchNormalizationLayer('Name','BN4')
    
    % block 1
    convolution2dLayer(3,128,'DilationFactor',2,'Padding','same','Name','conv5_1')
    convolution2dLayer(3,128,'DilationFactor',2,'Padding','same','Name','conv5_2')
    reluLayer('Name','relu5')
    batchNormalizationLayer('Name','BN5')
    
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
pxdsPred = semanticseg(test_plds,net,'MiniBatchSize', 64, 'WriteLocation','preds');

metrics = evaluateSemanticSegmentation(pxdsPred,test_plds);

save net


