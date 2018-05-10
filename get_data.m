%% functionname: function description
function [data] = get_data(source_data,m,n)
	data=zeros(m,n);
	fid=fopen(source_data);
	for i=1:m
		str=fgetl(fid);
		s=textscan(str,'%d','delimiter',',');
		s_length=length(s{1,1});
		sample_length=s_length/2;
		for j=1:sample_length
			%s{1,1}(j*2-1);
			data(i,s{1,1}(j*2-1)+1)=1;
		end
	end