function [ vector] = extract( z,ce ,label,wei)
vector1=[];
vector2=[];
vector=[];
for m=1:12
for i=1:label
    for j=1:wei-m
    vector1(i,j)=z{m}(ce{m}(i,j));

    end
end
vvv{m}=vector1;
vector=[vector,vector1];
vector1=[];
end
end

