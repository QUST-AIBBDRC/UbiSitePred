clear all
clc
A1=importdata('set1_positive.txt');
A2=importdata('set1_negative.txt');
wei=13;
[num1]=numel(A1);
train_shu1=[A1;A2];
train_label1=[ones(num1,1);ones(num1,1)*2];
[train_shu,train_label] = vector(train_shu1,train_label1,wei);
A=[train_shu,train_label];
save pspm1_11.mat A