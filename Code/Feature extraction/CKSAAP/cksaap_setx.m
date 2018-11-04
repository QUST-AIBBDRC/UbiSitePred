clear all
clc
input=importdata('Set1.txt');
label=input.data(:,2);
data=input.textdata(:,1);
num=numel(data);
sign0=find(label==0);
sign1=find(label==1);
sequence=[data(sign1,:);data(sign0,:)];
sequence_labele=[ones(6118,1);zeros(6118,1)];
out=[];
inputout=sequence;
for i=1:num 
    protein=inputout{i};
    output1 =Dipeptide(protein);%Call the Dipeptide function to calculate the cksaap value of each sequence
    out=[out;output1];
end
save cksaap1_7.mat out