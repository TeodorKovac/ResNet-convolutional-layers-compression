%
%imgFolder = '/Users/petrkovac/Public/ILSVRC2013/litetrain';
open resnet18.mat
net = ans.net
imgFolder = '/disk/kovac/Train';
fprintf('nahralo se TRAIN \n')
imds = imageDatastore(imgFolder,'IncludeSubfolders',true,'LabelSource','foldernames');
%net = resnet50;
inputSize = net.Layers(1).InputSize;

fprintf('split the database \n')
[imdsTrain,imdsTest] = splitEachLabel(imds, 0.7, 'randomize');

fprintf('resize the pictures to match the inputsize proper \n')
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain,  'ColorPreprocessing', 'gray2rgb');
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest,  'ColorPreprocessing', 'gray2rgb');

%% 

lgraph = layerGraph(net);
num_layers = size(lgraph.Layers,1);
classLayer = lgraph.Layers(num_layers);
softmaxlayer = lgraph.Layers(num_layers-1);
fullyconnected = lgraph.Layers(num_layers-2);

%% NEW
numClasses = numel(categories(imdsTrain.Labels));


new_fullyconnected = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);

new_softmaxlayer = softmaxLayer('Name','new_sft');

new_classLayer = classificationLayer('Name','new_classoutput');
%% REPLACE
lgraph = replaceLayer(lgraph,classLayer.Name,new_classLayer);
lgraph = replaceLayer(lgraph,softmaxlayer.Name,new_softmaxlayer);
lgraph = replaceLayer(lgraph,fullyconnected.Name,new_fullyconnected);

%% FREEZE WEIGHTS
%layers = lgraph.Layers;
%connections = lgraph.Connections;

%layers(1:(num_layers-3)) = freezeWeights(layers(1:(num_layers-3)));
%lgraph = createLgraphUsingConnections(layers,connections);
% 
%% TRAINING OPTIONS

miniBatchSize = 10;
valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'ExecutionEnvironment', 'gpu', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',3e-3, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsTest, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',true, ...
    'Plots','training-progress');

%% TRENINK
fprintf('GPU... \n')

fprintf('trainNetwork... \n')
net1 = trainNetwork(augimdsTrain,lgraph,options);
fprintf('finished. Test accuraccy: \n')
[YPred,scores]= classify(net1,augimdsTest);
YValidation = imdsTest.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)

save('proper_resnet18','net1')

%% PROJIT OBRAZKY












    
