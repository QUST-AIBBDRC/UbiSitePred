function [z] = multiplematrix( ce,set,label,wei )
for m=1:12
b=zeros(441,(wei-m));
a=ce{m};
for i=1:(wei-m)
    for j=1:label
        if a(j,i)==0
           continue;
        else
            ni=a(j,i);
            for k=1:label
                if ni==a(k,i)
                    b(ni,i)=b(ni,i)+1;
                    a(k,i)=0;
                end
            end
        end
    end
end
z{m}=b./label;
end
end




            
            
            
            

