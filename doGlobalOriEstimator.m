function [dblDecodedAngle] = doGlobalOriEstimator(vecThisAct, matThisData, vecThisOriRads)
	% Returns angles in radians
	
	%function [vecWeights] = ComplexEstimator(matThisData, vecThisOriRads)
	
	% matThisData must have dims [trials x neurons].
	% vecClasses must have dims [trials x 1].
	
	intTrials = size(matThisData,1);
	
	w = sum(exp(1i * vecThisOriRads).*matThisData,1)/intTrials;
	Sigma = matThisData'*matThisData/intTrials;
	
	vecWeights = Sigma\w;
	
	dblDecodedAngle = angle(vecThisAct * vecWeights);
end
