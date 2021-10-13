function [matWeights, dblLLH] = doMnLogReg(matData, vecTrialTypes, dblLambda)
	% Multiclass logistic regression
	% Input:
	%   matData: [neuron x trial] data matrix
	%   vecTrialTypes: [1 x trial] label
	%   dblLambda: regularization parameter
	% Output:
	%   matWeights: trained weights
	%   dblLLH: log-likelihood
	% By Jorrit Montijn
	
	if nargin < 3
		dblLambda = 0;
	end
	matData = [matData; ones(1,size(matData,2))];
	[matWeights, dblLLH] = newtonRaphson(matData, label2idx(vecTrialTypes), dblLambda);
end
function [matWeights, dblLLH] = newtonRaphson(matData, vecTrialTypes, dblLambda)
	[intNeurons,intTrials] = size(matData); %number of predictors (neurons) and observations (trials)
	intClasses = numel(unique(vecTrialTypes)); %number of classes in data
	
	%run
	intRespCombos = intNeurons*intClasses; %number of things to predict
	vecRespCombos = (1:intRespCombos)'; %indexing vector
	vecLinearRCs = sub2ind([intRespCombos,intRespCombos],vecRespCombos,vecRespCombos); %linear index vector
	spTargetClasses = sparse(vecTrialTypes,1:intTrials,1,intClasses,intTrials,intTrials); %sparse matrix containing the target classes (t=1) and non-target classes (t=0)
	matWeights = zeros(intNeurons,intClasses)./intClasses; %pre-allocate weight matrix; [intNeuron x intClass]
	matPredictUpdate = zeros(intNeurons,intClasses,intNeurons,intClasses); %pre-allocate iterative error matrix
	
	% Bishop Eq 4.105; determine the 'activation', or prediction, for each class, for each trial; [intClass x intTrial]
	matPrediction = matWeights'*matData;
	% Bishop Eq 4.104; soft-max operation to create posterior probabilities of the different classes
	matPosteriorProbabilities = bsxfun(@minus,matPrediction,logsumexp(matPrediction,1));
	% Bishop Eq 4.108; cross-entropy function with regularization lambda on the weights
	dblLLH = dot(spTargetClasses(:),matPosteriorProbabilities(:))-0.5*dblLambda*dot(matWeights(:),matWeights(:));
	
	%perform weight updates
	hTic=tic;
	matResidual = exp(matPosteriorProbabilities);
	for intClass1 = 1:intClasses
		if toc(hTic) > 5
			fprintf('Running class %d/%d [%s]\n',intClass1,intClasses,getTime);
			hTic=tic;
		end
		for intClass2 = 1:intClasses
			vecR_Class1Class2 = matResidual(intClass1,:).*((intClass1==intClass2)-matResidual(intClass2,:));  %[1 x intTrial], r has negative value, so cannot use sqrt
			matPredictUpdate(:,intClass1,:,intClass2) = bsxfun(@times,matData,vecR_Class1Class2)*matData';	%[intNeuron x intNeuron] difference in predictive value for class1 vs class2
		end
	end
	matWeightUpdate = matData*(matResidual-spTargetClasses)'+dblLambda*matWeights; %[intNeuron x intClass]
	matErrorTrialxTrial = reshape(matPredictUpdate,intRespCombos,intRespCombos); %[(intNeuron x intClass) x (intNeuron x intClass)]
	matErrorTrialxTrial(vecLinearRCs) = matErrorTrialxTrial(vecLinearRCs)+dblLambda; %add regularization penalty
	matWeights(:) = matWeights(:)-matErrorTrialxTrial\matWeightUpdate(:); %update weights by error
end
