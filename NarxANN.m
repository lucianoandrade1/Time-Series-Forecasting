%A Nonlinear Autoregressive Exogenous (NARX) Neural Network
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
Matrix = TdnnMatrix(Z, 3, 1, 4);

[A B] = size(Matrix);
C = A - round(A/7);

trnData = Matrix(1:C,1:4)';
trnDataOut = Matrix(1:C,5)';
ChkData = Matrix(C:A,1:4)';
OutDes = Matrix(C:A,5)';

narx_net = newnarx(trnData,trnDataOut,[0 5]);
narx_net.trainParam.show = 10;
narx_net.trainParam.epochs = 20;

narx_net = train(narx_net,trnData, trnDataOut);

out = sim(narx_net, ChkData);

OutDesR = mapminmax('reverse', OutDes, PS);
outR = mapminmax('reverse', out, PS);

figure;
plot(OutDesR./1000000, 'b', 'LineWidth',2);
hold on;
plot(outR./1000000, 'r', 'LineWidth',2);
ylabel('Load MW');
h = legend('Desired','Obtained', 2);


APEs = (OutDesR-outR)./OutDesR;

figure;
bar(APEs.*100);
ylabel('Percent error');

figure;
hist (APEs.*100);
xlabel('Percent Relative Error');
title('Error Histogram')

APEm = max(abs(APEs))*100
[a b] = size(outR);
MAPE = (sum(abs((OutDesR - outR)./OutDesR))/b)*100
Variance = sum((APEs - mean(APEs)).^2)/b


