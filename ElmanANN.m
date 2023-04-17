%Elman-RNN: Elman Recurrent Neural Network (RNN)
%Load demand forecasting source code implemented in Matlab 2010
%Author: Luciano Andrade
%Unfortunately I was not allowed to share the time series data used

clear all;
clc;

Z = load('Every5MinutesVot.txt');
%Z = load('Every5MinutesUba.txt');
%Z = load('Every5MinutesAnd.txt');
%Z = load('Every5MinutesMog.txt');
%Z = load('Every5MinutesCor.txt');

[Y,PS] = mapminmax(Z');
Z = Y';
Matrix = TdnnMatrix(Z, 2, 1, 288)

[A B] = size(Matrix);
C = A - round(A/9);

trnData = Matrix(1:C,1:3)';
trnDataOut = Matrix(1:C,4)';
ChkData = Matrix(C:A,1:3)';
OutDes = Matrix(C:A,4)';

net = newelm(minmax(trnData),[3 1],{'tansig', 'purelin'},'trainlm','learngdm','mse');
net.trainParam.goal =  1e-7;
net.trainParam.show =  1;
net.trainParam.epochs = 15; 
net = train(net, trnData, trnDataOut)

out = sim(net,ChkData);

OutDesR = mapminmax('reverse', OutDes, PS);
outR = mapminmax('reverse', out, PS);

figure;
plot(C:A,OutDesR,'--','LineWidth',2);
hold on;
plot(C:A,outR,'r-','LineWidth',2); hold off;
xlabel('Time (every 5 minutes)');
ylabel('Load W');

APEs = (OutDesR-outR)./OutDesR;

figure;
hist (APEs);
xlabel('Relative Error');
title('Error Histogram')

[a b] = size(outR)
MAPE = (sum(abs((OutDesR - outR)./OutDesR))/b)*100
Variance = sum((APEs - mean(APEs)).^2)/b


