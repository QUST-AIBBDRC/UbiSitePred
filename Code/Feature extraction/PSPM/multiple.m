function [ce] =multiple( set,label,wei)
s=[];
for k=1:13
for j=1:(wei-k)
    for i=1:label
   s(i,j)= 21*(set(i,j)-1)+set(i,j+k);
    end
end
ce{k}=s;
s=[];
end
end