%Time delay matrix funtion
%Load demand forecasting source code implemented in Matlab 2010
%Author: Luciano Andrade
%Function implemented to create the inputs and outputs matrix to load
%   demand forecasting
function [Matrix] = TdnnMatrix(Z, window, NInput, PUnit)
clc;

[M,N] = size(Z);
v = zeros(M,window + 1 + NInput);
for i=window:1:M
   k = i;
   for j=1:1:window
     v(i-window+1,j) = Z(k);
     k = k - 1;
   end
   if NInput>0
       n=NInput;
       for l=window:1:window+NInput
           v(i - window + n*PUnit,l) = Z(i);
           n=n-1;       
           if n==0
              n=NInput; 
           end
       end
   end
end

for i=2:M-window+1
    v(i-1,window+1+NInput) = v(i,1);
end

Matrix = v(NInput*PUnit+1:M-window,1:window+1+NInput);

