function [train_shu,train_label ] = vector(train_shu1,train_label1,wei )
  z1=find(train_label1==1);
  f1=find(train_label1==2);
  label=min(length(z1),length(f1));
  set1=ertaimatrix(train_shu1(z1,:),label,wei);
  set2=ertaimatrix(train_shu1(f1,:),label,wei);
  ce1=multiple(set1,label,wei);
  ce2=multiple(set2,label,wei);
  z1=multiplematrix( ce1,set1,label,wei);
  z2=multiplematrix( ce2,set2,label,wei);
   for m=1:12
   z{m}=z1{m}-z2{m};
   end
   vector1= extract( z,ce1,label,wei);
   vector2= extract( z,ce2,label,wei);
   train_shu=[ vector1;vector2];
   train_label=[ones(label,1);ones(label,1)*2];
end

