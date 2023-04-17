%ANFIS: Adaptive Neuro Fuzzy Inference System
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
Matrix = TdnnMatrix(Z, 2, 1, 288);
[A B] = size(Matrix);

C = A - 2*round(A/7);
D = A - round(A/7);

trnData = Matrix(1:C,1:4);
ChkData = Matrix(C:D,1:4);
%ChkDataOut = Matrix(C:A,4);
TestData = Matrix(D:A,1:3);
TestDataOut = Matrix(D:A,4);


numMFs = [3 3 3];
epoch_n = 350;
 
in_fis = genfis1(trnData, numMFs, 'gbellmf','constant')
[out_fis, trn_error, ss, out_fis1, chk_error] = anfis(trnData,in_fis,epoch_n,[],ChkData);

outDes = TestDataOut;
out = evalfis(TestData,out_fis1);

figure;
plot(trn_error, '--', 'LineWidth',2);
hold on; 
plot(chk_error, 'r-','LineWidth',2); hold off;
xlabel('Epochs');
ylabel('MSE (Mean Square Error)');
title('Error curves');

h = legend('Trainning','Validation',1);
set(h,'Interpreter','none')

outDesR = mapminmax('reverse', outDes, PS);
outR = mapminmax('reverse', out, PS);

outDesX = outDesR./1000000
outX = outR./1000000

z = (D-A)/24

figure;
plot(D:A,outDesX,'--','LineWidth',2); % axis ([D A 3.8 10.5]); %axis ([D A 0 8]); %axis ([D A 10 14.1]);
hold on;
plot(D:A,outX,'r-','LineWidth',2); hold off;
xlabel('Measures (every 5 minutes)');
ylabel('Load MW');
set(gca,'FontName','times','FontSize', 10, 'xtick', [D D+10.7 D+21.4 D+32.1 D+42.8 D+53.5 D+64.2 D+74.9 D+85.6 D+96.3 D+107 D+117.7 D+128.4 D+139.1 D+149.8 D+160.5 D+171.2 D+181.9 D+192.6 D+203.3 D+214 D+224.7 D+235.4 D+246.1 D+256]);
set(gca,'XTickLabel',{'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24'});

h = legend('Measured Load','Forecasted Load',2);
set(h,'Interpreter','none')

MAPE = (outDesR-outR)./outDesR;

[a b] = size(MAPE)
line1 = zeros(a);

figure;
plot (MAPE,'o','LineWidth',2); hold on;
plot(line1,'b:');
set(findobj(gca,'Type','line','Color',[0 0 1]), 'Color','black','LineWidth',2)
h = legend('Relative Error',3);

x = -0.06:0.01:0.06;
figure;
hist (MAPE, x);
xlabel('Relative Error');
title('Error Histogram');
h = findobj(gca,'Type','patch');
set(h,'FaceColor',[0.5 0.5 0.5],'EdgeColor','k');

avgMAPE = (mean(abs((outDesR-outR)./outDesR)))*100
Variance = sum((MAPE - mean(MAPE)).^2)/a
