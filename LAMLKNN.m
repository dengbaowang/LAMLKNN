function [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision]=MLKNN(train_data,train_target,test_data,test_target,Num,K,Smooth)
	% This is a extension of MLkNN algorithm which was proposed by Lamda group.
	% Inputs:
	%   train_data      - A n x m array, The i-th instance is stored in train_data(i,:)
	%   train_target    - A n x T array, T is the number of possible labels, train_target(i,j) is 1 if the i-th instance has the j-th label, and -1 otherwise
	%   test_data       - A n_test x m array, n_test is the number of test instances
	%   test_target     - A n_test x T array
	%   Num 			- k value of kNN algorithm
	% 	K 				- number of Cluster
	%	Smooth			- parameter of Laplace Smoothing Term

	tic
	[num_cluster,centersK,PriorK,Prior,PriorN,Cond,CondN]=MLKNN_train(train_data,train_target,Num,K,Smooth);
	[HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels]=MLKNN_test(train_data,train_target,test_data,test_target,Num,centersK,num_cluster,PriorK,Prior,PriorN,Cond,CondN);
	toc