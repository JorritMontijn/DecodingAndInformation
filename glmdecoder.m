function [matWeights,matGoodX,matGoodAct] = glmdecoder(matAct,matFactors,matX)
	%glmdecoder Build weight matrix for glm feature decoding
	%   [matWeights,matGoodX,matGoodAct] = glmdecoder(matAct,matFactors,matX)
	%
	%matAct: activity matrix [time x neurons]
	%matFactors: model factors [neurons x features]
	%matX: real features corresponding to [time x features]
	%
	%Decode using: matDecodedX = matAct * matWeights * matFactors

	%make factors global
	global matGlobF;
	matGlobF = matFactors(:,2:end);
	
	%remove nans from matX
	indGood = ~any(isnan(matX),2);
	matGoodX = matX(indGood,:);
	matGoodAct = matAct(indGood,:);
	
	%find optimal weight matrix
	intN = size(matGoodAct,2);
	funcFit = @(vecWeights,matAct) matAct*diag(vecWeights)*matGlobF;
	vecWeights0 = ones(1,intN);
	matWeights = diag(lsqcurvefit(funcFit, vecWeights0, matGoodAct, matGoodX,0*vecWeights0,intN*vecWeights0));
end

