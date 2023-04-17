%FF-ANN: Feed forward Artificial Neural Network
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
C = A - round(A/7);

trnData = Matrix(1:C,1:3)';
trnDataOut = Matrix(1:C,4)';
ChkData = Matrix(C:A,1:3)';
OutDes = Matrix(C:A,4)';

net = newff(minmax(trnData),[3 1],{'tansig', 'purelin'},'trainlm');
net.trainParam.goal =  1e-10;
net.trainParam.show =  1;
net.trainParam.epochs = 100; 
net = train(net, trnData, trnDataOut)

out = sim(net,ChkData);

OutDesR = mapminmax('reverse', OutDes, PS);
outR = mapminmax('reverse', out, PS);


figure;
plot(C:A,OutDesR/1000000,'--','LineWidth',2); axis ([C A 5 15]); 
hold on;
plot(C:A,outR/1000000,'r-','LineWidth',2); hold off;
xlabel('Medidas (intervalos de 5 minutos)');
ylabel('Carga MW');
set(gca,'FontName','times','FontSize', 10, 'xtick', [C C+10.25 C+20.5 C+30.75 C+41 C+51.25 C+61.5 C+71.75 C+82 C+92.25 C+100.5 C+110.75 C+122 C+132.25 C+142.5 C+152.75 C+163 C+173.25 C+183.5 C+193.75 C+204 C+214.25 C+224.5 C+234.75 C+245]);
set(gca,'XTickLabel',{'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24'});

h = legend('Carga Mensurada','Carga Prevista',2);
set(h,'Interpreter','none')

APEs = (OutDesR-outR)./OutDesR;

figure;
hist (APEs);
xlabel('Relative Error');
title('Error Histogram')

[a b] = size(outR)
MAPE = (sum(abs((OutDesR - outR)./OutDesR))/b)*100
Variance = sum((APEs - mean(APEs)).^2)/b


