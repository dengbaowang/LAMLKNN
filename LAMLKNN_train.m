function [num_cluster,centersK,PriorK,Prior,PriorN,Cond,CondN]=MLKNN_train(train_data,train_target,Num,K,Smooth)
%MLKNN_train trains a multi-label k-nearest neighbor classifier
%
%    Syntax
%
%       [Prior,PriorN,Cond,CondN]=MLKNN_train(train_data,train_target,num_neighbor)
%
%    Description
%
%       KNNML_train takes,
%           train_data   - An MxN array, the ith instance of training instance is stored in train_data(i,:)
%           train_target - A QxM array, if the ith training instance belongs to the jth class, then train_target(j,i) equals +1, otherwise train_target(j,i) equals -1
%           Num          - Number of neighbors used in the k-nearest neighbor algorithm
%           Smooth       - Smoothing parameter
%      and returns,
%           Prior        - A Qx1 array, for the ith class Ci, the prior probability of P(Ci) is stored in Prior(i,1)
%           PriorN       - A Qx1 array, for the ith class Ci, the prior probability of P(~Ci) is stored in PriorN(i,1)
%           Cond         - A Qx(Num+1) array, for the ith class Ci, the probability of P(k|Ci) (0<=k<=Num) i.e. k nearest neighbors of an instance in Ci will belong to Ci , is stored in Cond(i,k+1)
%           CondN        - A Qx(Num+1) array, for the ith class Ci, the probability of P(k|~Ci) (0<=k<=Num) i.e. k nearest neighbors of an instance not in Ci will belong to Ci, is stored in CondN(i,k+1)

    [num_class,num_training]=size(train_target);
    [~,num_features]=size(train_data);
    %D=pdist([zeros(1,num_features);ones(1,num_features)],'euclidean')*K/num_training
    tic
%Computing distance between training instances
    dist_matrix=diag(realmax*ones(1,num_training));
    for i=1:num_training-1
        if(mod(i,100)==0)
            disp(strcat('computing distance for instance:',num2str(i)));
        end
        vector1=train_data(i,:);
        for j=i+1:num_training            
            vector2=train_data(j,:);
            dist_matrix(i,j)=sqrt(sum((vector1-vector2).^2));
            dist_matrix(j,i)=dist_matrix(i,j);
        end
    end

    Neighbors=cell(num_training,1); %Neighbors{i,1} stores the Num neighbors of the ith training instance
    for i=1:num_training
        [temp,index]=sort(dist_matrix(i,:));
        Neighbors{i,1}=index(1:Num);
    end
    near_num_label=zeros(num_training,num_class);
    for i=1:num_training
        temp=zeros(1,num_class); %The number of the Num nearest neighbors of the ith instance which belong to the jth instance is stored in temp(1,j)
        neighbor_labels=[];
        for j=1:Num
            neighbor_labels=[neighbor_labels,train_target(:,Neighbors{i,1}(j))];
        end
        for j=1:num_class
            near_num_label(i,j)=sum(neighbor_labels(j,:)==ones(1,Num));
        end
    end
%K-means for traning data
    %trainofK=zeros(T*K,num_training);
    %centersK=zeros(T*K,num_features);
    num_cluster=0;
    tic
    [WhichK,centers,sum1]=kmeans([train_data],K,'emptyaction','singleton');
    %sumd=sum(sum1)
    centers=centers(:,1:num_features);
    for c=1:K
        temp_here=find(WhichK==c)';
        temp_count=sum([d,~]=find(temp_here));
        if(1)%temp_count>=num_training*0.6/K&&temp_count<=num_training*1.4/K)
            num_cluster++;
            trainofK(num_cluster,1:length(find(WhichK==c)))=temp_here;
            centersK(num_cluster,:)=centers(c,:);
            numK(num_cluster)=temp_count;
            PriorK(num_cluster)=numK(num_cluster)/num_training;
        end
    end
    toc
%Computing Prior and PriorN
    Prior=zeros(num_class,num_cluster);
    PriorN=zeros(num_class,num_cluster);
    for c=1:num_cluster
        %ind=find(WhichK==c);
        [~,ind_temp]=find(trainofK(c,:));
        ind=trainofK(c,ind_temp);
        for i=1:num_class
            temp_Ci=sum(train_target(i,ind)==ones(1,numK(c)));
            
            Prior(i,c)=(Smooth+temp_Ci)/(Smooth*2+numK(c));
            PriorN(i,c)=1-Prior(i,c);
        end
    end
%Computing Cond and CondN
    Neighbors=cell(num_training,1); %Neighbors{i,1} stores the Num neighbors of the ith training instance
    for i=1:num_training
        [temp,index]=sort(dist_matrix(i,:));
        Neighbors{i,1}=index(1:Num);
    end
    
    
    Cond=zeros(num_class,Num,num_cluster);
    CondN=zeros(num_class,Num,num_cluster);
    for c=1:num_cluster
        temp_Ci=zeros(num_class,Num+1); %The number of instances belong to the ith class which have k nearest neighbors in Ci is stored in temp_Ci(i,k+1)
        temp_NCi=zeros(num_class,Num+1); %The number of instances not belong to the ith class which have k nearest neighbors in Ci is stored in temp_NCi(i,k+1)
        [~,ind_temp]=find(trainofK(c,:));
        ind=trainofK(c,ind_temp);
        for i=ind
            temp=zeros(1,num_class); %The number of the Num nearest neighbors of the ith instance which belong to the jth instance is stored in temp(1,j)
            neighbor_labels=[];
            for j=1:Num
                neighbor_labels=[neighbor_labels,train_target(:,Neighbors{i,1}(j))];
            end
            for j=1:num_class
                temp(1,j)=sum(neighbor_labels(j,:)==ones(1,Num));
            end
            for j=1:num_class
                if(train_target(j,i)==1)
                    temp_Ci(j,temp(j)+1)=temp_Ci(j,temp(j)+1)+1;
                else
                    temp_NCi(j,temp(j)+1)=temp_NCi(j,temp(j)+1)+1;
                end
            end
        end
        for i=1:num_class
            temp1=sum(temp_Ci(i,:));
            temp2=sum(temp_NCi(i,:));
            for j=1:Num+1
                Cond(i,j,c)=(Smooth+temp_Ci(i,j))/(Smooth*(Num+1)+temp1);
                CondN(i,j,c)=(Smooth+temp_NCi(i,j))/(Smooth*(Num+1)+temp2);
            end
        end  
    end            
    toc