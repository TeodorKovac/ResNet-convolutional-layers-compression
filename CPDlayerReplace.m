function [net1,tmp_net,sens,err] = CPDlayerReplace(net,layerNum,Rank,it,mez1)
%% 
% FUNKCE, ktera zameni vrstvu s cislem layerNum rozkladem pomoci KLM

%% PARAMETRY
R = Rank;

T = net.Layers(layerNum).Weights;
T = double(T);
%%
PaddingSize = net.Layers(layerNum).PaddingSize;
Stride = net.Layers(layerNum).Stride;
NoGroups = size(T,5);
NoOutChannels = size(T,4)*NoGroups;
NoInChannels = size(T,3);
FilterSize = [size(T,1) size(T,2)];

%%
inputSize = net.Layers(1).InputSize;

%% ROZKLAD KLM
numit = it;
mez = mez1;
m = 2;
fprintf('Probiha KLM \n')
[A,B,C,D,iter]=KLM4cRE(T,R,mez,m,numit);
sens = sensitivity(A,B,C,D);
fprintf('Sensitivita: %f \n',sens)
fprintf('Rozklad s errorem: \n')
iter(end)
err = iter(end)/(size(T,1)*size(T,2)*size(T,3)*size(T,4));
%%  PREPOJENI VRSTEV
tmp_net = layerGraph(net);

conv_1 = convolution2dLayer(1,R,'Padding',0,'BiasLearnRateFactor',0,'Name',['conv_',num2str(layerNum),'_1'],'Weights',single(reshape(C,1,1,[],R)),'NumChannels',NoInChannels,'Bias',zeros(1,1,R));
conv_2 = groupedConvolution2dLayer([FilterSize(1), 1], 1, 'channel-wise', 'Padding',[PaddingSize(1) 0],'BiasLearnRateFactor',0,'Stride', [Stride(1) 1], 'Name',['conv_',num2str(layerNum),'_2'],'Weights',single(reshape(A,[],1,1,1,R)),'Bias',zeros(1,1,1,R));
conv_3 = groupedConvolution2dLayer([1 FilterSize(2)],1,'channel-wise','Padding',[0 PaddingSize(3)],'BiasLearnRateFactor',0,'Stride',[1 Stride(2)],'Name',['conv_',num2str(layerNum),'_3'], 'Weights',single(reshape(B,1,[],1,1,R)),'Bias',zeros(1,1,1,R));
conv_4 = convolution2dLayer([1 1],NoOutChannels,'Padding',0,'Name',['conv_',num2str(layerNum),'_4'], 'Weights',single(reshape(D',1,1,R,[],1)),'Bias',reshape(tmp_net.Layers(layerNum).Bias,1,1,[]),'NumChannels',R);

tmp_net = addLayers(tmp_net,conv_1);
tmp_net = addLayers(tmp_net,conv_2);
tmp_net = addLayers(tmp_net,conv_3);
tmp_net = addLayers(tmp_net,conv_4);

connect_name1 = tmp_net.Layers(layerNum-1).Name; %jmeno vrstvy na kterou pripojim
connect_name2 = tmp_net.Layers(layerNum+1).Name; %jmeno vrstvy ke ktere pripojim
old_name = tmp_net.Layers(layerNum).Name; %jmeno stare vrstvy
numLayers = size(tmp_net.Layers,1);

%Pripojeni 4 vrstev
tmp_net = connectLayers(tmp_net,char(net.Connections{find(strcmp(old_name,[net.Connections{:,2}])),1}),tmp_net.Layers(numLayers-3).Name);
tmp_net = connectLayers(tmp_net,tmp_net.Layers(numLayers-3).Name, tmp_net.Layers(numLayers-2).Name);
tmp_net = connectLayers(tmp_net,tmp_net.Layers(numLayers-2).Name, tmp_net.Layers(numLayers-1).Name);
tmp_net = connectLayers(tmp_net,tmp_net.Layers(numLayers-1).Name, tmp_net.Layers(numLayers).Name);

%Odpojeni stare vrstvy
tmp_net = disconnectLayers(tmp_net,char(net.Connections{find(strcmp(old_name,[net.Connections{:,2}])),1}),old_name);
tmp_net = disconnectLayers(tmp_net,old_name, char(net.Connections{find(strcmp(old_name,[net.Connections{:,1}])),2}));
%Pripojeni posledni
tmp_net = connectLayers(tmp_net,tmp_net.Layers(numLayers).Name, char(net.Connections{find(strcmp(old_name,[net.Connections{:,1}])),2}));

%Odstraneni stare vrstvy
tmp_net = removeLayers(tmp_net,old_name);

plot(tmp_net);
%% ZAPSANI HODNOT
netX = assembleNetwork(tmp_net);

net1 = netX;

end