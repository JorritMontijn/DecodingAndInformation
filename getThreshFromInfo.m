function vecDiscriminationThreshold = getThreshFromInfo(vecFisher)
	%getThreshFromInfo Gives discrimination thresholds from Fisher I
	%   vecDiscriminationThreshold = getThreshFromInfo(vecFisher)
	
	vecDiscriminationThreshold = 1./sqrt(vecFisher); %Kanitscheider (2015)
end

