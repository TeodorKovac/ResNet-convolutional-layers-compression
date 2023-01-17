
open resnet18_transferLearning.mat % resnet18 po transfer learning
net1 = ans.net1;

imgFolder = '/disk/kovac/Train'; % adresa databáze
imds = imageDatastore(imgFolder,'IncludeSubfolders',true,'LabelSource','foldernames');
inputSize = net1.Layers(1).InputSize;

[imdsTrain,imdsTest] = splitEachLabel(imds, 0.7, 'randomize');

% změna velikosti jednotlivých obrázků
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain,  'ColorPreprocessing', 'gray2rgb');
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest,  'ColorPreprocessing', 'gray2rgb');
%% POCATECNI ACCURACY
[YPred,scores]= classify(net1,augimdsTest);
YValidation = imdsTest.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation);

P = [2 6 9 13 16 20 23 29 32 36 39 45 48 52 55 61 64]; % pořadí jednotlivých konvolučních vrstev v síti

Acc = zeros(1,size(P,2)+1); 
Top5Acc = zeros(1,size(P,2)+1);
Acc(1) = accuracy; % původní Top 1 přesnost
it = 1000;

%% ZAMĚNOVÁNÍ VRSTEV

Rank = (2)*[30 38 38 38 38 46 62 62 62 82 120 120 120 160 236 236 236]; % hodnost pro jednotlivé CP rozklady
mez = [500 500 500 500 500 500 700 700 700 1000 1400 1400 1400 2250 2700 2700 2700]; % Sensitivita pro jednotlivé CP rozklady
Sensitivita = zeros(1,size(P,2));
err = zeros(size(Rank));
net2 = net1;
for i=1:size(P,2)

    [net2,tmp_net,Sensitivita(i),err(i)] = CPDlayerReplace(net2,(P(i)-(i-1)),Rank(i),it,mez(i)); % volání funkce zaměníjucí konvoluční jádro za CP rozklad
    
    %fine-tuning
    
    miniBatchSize = 10;
    valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
    options = trainingOptions('sgdm', ...
        'ExecutionEnvironment', 'gpu', ...
        'MiniBatchSize',miniBatchSize, ...
        'MaxEpochs',2, ...
        'InitialLearnRate',3e-3, ...
        'Shuffle','every-epoch', ...
        'ValidationData',augimdsTest, ...
        'ValidationFrequency',valFrequency, ...
        'Verbose',true, ...
        'Plots','training-progress');
    [net2,info] = trainNetwork(augimdsTrain,tmp_net,options);
    [YPred,scores]= classify(net2,augimdsTest,'ExecutionEnvironment','cpu');
    YValidation = imdsTest.Labels;

    accuracy = sum(YPred == YValidation)/numel(YValidation);
    Acc(i+1) = accuracy;
    
    %% TOP5 
[n,m] = size(scores);  
idx = zeros(m,n); 
for k=1:n  
    [~,idx(:,k)] = sort(scores(k,:),'descend');  
end  
idx = idx(1:5,:);  
top5Classes = net2.Layers(71-i).ClassNames(idx);  
top5count = 0;  
for k = 1:n  
    top5count = top5count + sum(YValidation(k,1) == top5Classes(:,k));  
end  
Top5Acc(i+1) = top5count/n; 
    
    save('Sensitivita', 'Sensitivita','Acc','Top5Acc','err','net2');
    
end 
