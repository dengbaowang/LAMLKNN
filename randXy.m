function [X_rand,y_rand]=randXy(rand,X,y)
for i=1:2000
  X_rand(i,:)=X(rand(i),:);
  y_rand(:,i)=y(:,rand(i));
  end