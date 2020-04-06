function [matWeights,matGoodX,matGoodAct] = lineardecoder(matAct,matX)
	%lineardecoder Build weight matrix for linear feature decoding
	%   [matWeights,matGoodX,matGoodAct] = lineardecoder(matAct,matX)
	%
	%matAct: activity matrix [time x neurons]
	%matFactors: model factors [neurons x features]
	%matX: real features corresponding to [time x features]
	%
	%Decode using: matDecodedX = matAct * matWeights * matFactors
	%
	%	Version History:
	%	2019-04-08	Created by Jorrit Montijn
	
	%remove nans from matX
	indGood = ~any(isnan(matX),2);
	matGoodX = matX(indGood,:);
	matGoodAct = matAct(indGood,:);
	
	%find optimal weight matrix
	intFactors = size(matX,2);
	intN = size(matGoodAct,2);
	
	%define function
	funcFit = @(vecWeights,matAct) vecWeights(1) + vecWeights(2)*matAct*vecWeights(3:end)';
	
	%% build bias-scaling-mean model
	%go through factors
	matWeights = nan(intN+2,intFactors);
	for intFactor=1:intFactors
		vecWeights0 = [-mean(matX(:,intFactor)) ones(1,intN+1)];
		matWeights(:,intFactor) = curvefitfun(funcFit, vecWeights0, matGoodAct, matGoodX(:,intFactor),-10000*ones(size(vecWeights0)),10000*ones(size(vecWeights0)));
	end
end

