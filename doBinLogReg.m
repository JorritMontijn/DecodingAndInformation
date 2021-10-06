function [vecWeights, dblLLH, vecBinaryPrediction, vecPrediction] = doBinLogReg(matData, vecClasses, dblLambda)
	% Binary (two-class) logistic regression
	% Input:
	%   matData: [neuron x trial] data matrix
	%   vecTrialTypes: [1 x trial] label
	%   dblLambda: regularization parameter
	% Output:
	%   vecWeights: weights
	%   dblLLH: log-likelihood
	%	vecBinaryPrediction
	%	vecPrediction
	% By Jorrit Montijn
	
	
	if nargin < 3 || isempty(dblLambda)
		dblLambda = 0; %regularization parameter
	end
	
	%prep vars
	[vecIdx,vecUnique,vecCounts,cellSelect,vecRepetition] = val2idx(vecClasses(:));
	vecClassIdx = vecIdx(:)' - 1; %class index should be 0 or 1
	if numel(vecClassIdx) == size(matData,1)
		matData = matData';
	end
	matData = [matData; ones(1,size(matData,2))]; %add row of ones for intercepts
	[intNeurons,intTrials] = size(matData); %number of predictors (neurons) and observations (trials)
	intClasses = numel(unique(vecClasses)); %number of classes in data
	
	%run
	vecIdx = (1:intNeurons)'; %index vector
	vecLinearRCs = sub2ind([intNeurons,intNeurons],vecIdx,vecIdx); %linear index vector
	vecH = ones(1,intTrials);
	vecH(vecClassIdx==0) = -1; %transform vecH to [-1 1] so we can use the sign to differentiate classes
	vecWeights = zeros(intNeurons,1);
	vecPrediction = vecWeights'*matData;
	
	%do logistic regression
	vecActivation = sigmoid(vecPrediction);  
	vecR = vecActivation.*(1-vecActivation);  
	Xw = bsxfun(@times, matData, sqrt(vecR));
	matH = Xw*Xw';  
	matH(vecLinearRCs) = matH(vecLinearRCs)+dblLambda;
	matU = chol(matH); %cholesky decomposition
	vecG = matData*(vecActivation-vecClassIdx)'+dblLambda.*vecWeights;   
	vecP = -matU\(matU'\vecG);                        
	vecWeights = vecWeights+vecP;
	vecPrediction = vecWeights'*matData;
	dblLLH = -sum(log1pexp(-vecH.*vecPrediction))-0.5*sum(dblLambda.*vecWeights.^2);
	%get binary prediction of labels
	vecBinaryPrediction = vecPrediction;
	vecBinaryPrediction(vecPrediction<0)=vecUnique(1);
	vecBinaryPrediction(~(vecPrediction<0))=vecUnique(2);
	
end
