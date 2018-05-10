%% functionname: function description
function [data1] = min_max(data)
	[num_sample,num_feature]=size(data)
	for j=1:num_feature
		data1(:,j)=(data(:,j)-min(data(:,j)))/(max(data(:,j))-min(data(:,j)));
	end