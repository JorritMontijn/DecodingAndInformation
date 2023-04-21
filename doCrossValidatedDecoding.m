function [dblPerformanceCV,vecDecodedIndexCV,matPosteriorProbability,dblMeanErrorDegs,matConfusion] = ...
		doCrossValidatedDecoding(matData,vecTrialTypes,intTypeCV,vecPriorDistribution,dblLambda)
	%doCrossValidatedDecoding Linear multivariate Gaussian decoder
	%[dblPerformanceCV,vecDecodedIndexCV,matPosteriorProbability,dblMeanErrorDegs,matConfusion] = ...
	%	doCrossValidatedDecoding(matData,vecTrialTypes,intTypeCV,vecPriorDistribution,dblLambda)
	%
	%Inputs:
	% - matData; [n x p]  Matrix of n observations/trials of p predictors/neurons
	% - vecTrialTypes; [n x 1] Trial indexing vector of c classes in n observations/trials
	% - intTypeCV; [int or vec] Integer switch 0-2 or trial repetition vector.
	%				Val=0, no CV; val=1, leave-one-out CV, val=2 leave-repetition-out.
	%				Val=vector, specifies fold indices for k-fold CV
	% - vecPriorDistribution: (optional) vector specifying # per trial type
	% - dblLambda; [scalar] "Regularization": 0=full MVN, inf=independent naive Bayes, in-between=mix
	%
	%Outputs:
	% - dblPerformance; [scalar] Fraction of correct classifications
	% - vecDecodedIndexCV; [n x 1] Decoded trial index vector of n observations/trials
	% - matPosteriorProbability; posterior probabilities
	% - dblMeanErrorDegs; [scalar] If vecTrialTypes is in radians, error in degrees
	% - matConfusion; [c x c] Confusion matrix of [(decoded class) x (real class)]
	%
	%Version History:
	%2023-04-20 Created function [by Jorrit Montijn]
	
	%% check which kind of cross-validation
	if ~exist('intTypeCV','var') || isempty(intTypeCV)
		intTypeCV = 2;
	end
	if ~exist('dblLambda','var') || isempty(dblLambda)
		dblLambda = 0;
	end
	if numel(dblLambda) ~= 1 || dblLambda < 0
		error([mfilename ':LambdaError'],'Lambda must be non-negative scalar');
	end
	
	%prior distribution
	if ~exist('vecPriorDistribution','var')
		vecPriorDistribution = [];
	end
	
	%% prepare
	intVerbose = 0;
	
	%get number of trials
	if ~all(isint(vecTrialTypes)) && range(vecTrialTypes) <= (2*pi)
		vecTrialTypes = rad2deg(vecTrialTypes);
	end
	vecTrialTypes = vecTrialTypes(:);
	intTrials = numel(vecTrialTypes);
	
	%check if matData is [trial x neuron] or [neuron x trial]
	if size(matData,1) == intTrials && size(matData,2) == intTrials
		%number of neurons and trials is the same
		warning([mfilename ':SameNeuronsTrials'],'Number of neurons and trials is identical; please double check the proper orientation of [intNeurons x intTrials]');
	elseif size(matData,1) == intTrials
		%rotate
		matData = matData';
	elseif size(matData,2) == intTrials
		%size is correct
	else
		error([mfilename ':SameNeuronsTrials'],'Size of matData and vecTrialTypes do not match');
	end
	intNeurons = size(matData,1);
	[vecTrialTypeIdx,vecUniqueTrialTypes,vecCounts,cellSelect,vecRepetition] = val2idx(vecTrialTypes);
	intStimTypes = length(vecUniqueTrialTypes);
	intRepNum = min(vecCounts);
	if ~isempty(vecPriorDistribution) && (numel(vecPriorDistribution) ~= intStimTypes || sum(vecPriorDistribution) ~= intTrials)
		error([mfilename ':MismatchPriorStimtypes'],'Size of vecPriorDistribution and vecTrialTypes do not match');
	end
	
	%pre-allocate output
	matPosteriorProbability = zeros(intStimTypes,intTrials);
	
	ptrTic = tic;
	%% cross-validate
	if numel(intTypeCV) == intTrials
		%designated sets
		
		%third input is fold index
		vecFoldIdx = intTypeCV;
		intFoldNum = max(vecFoldIdx);
		
		for intFold=1:intFoldNum
			%msg
			if toc(ptrTic) > 5
				ptrTic = tic;
				pause(eps);
				if intVerbose > 0,fprintf('Decoding; now at fold %d/%d [%s]\n',intFold,intFoldNum,getTime);end
			end
			
			%split trials
			indSelect = true(1,intTrials);
			indSelect(vecFoldIdx==intFold) = false;
			matTrainData = matData(:,indSelect);
			vecTrainTrialType = vecTrialTypeIdx(indSelect);
			matTestData = matData(:,~indSelect);
			
			%% calculate test probabilities by fitting a multivariate gaussian to the training data
			matTestPosterior = doMvnDec(matTrainData,vecTrainTrialType,matTestData,dblLambda);
			%assign output
			matPosteriorProbability(:,~indSelect) = matTestPosterior;
		end
		
	elseif intTypeCV == 0
		%no CV
		matPosteriorProbability = doMvnDec(matData,vecTrialTypeIdx,matData,dblLambda)';
			
	elseif intTypeCV == 1
		%leave one out
		for intLeaveOut=1:intTrials
			%get info on to-be-left-out trial
			indSelect = true(1,intTrials);
			indSelect(intLeaveOut) = false;
			
			%split trials
			matTrainData = matData(:,indSelect);
			vecTrainTrialType = vecTrialTypeIdx(indSelect);
			matTestData = matData(:,~indSelect);
			
			%% calculate test probabilities by fitting a multivariate gaussian to the training data
			matTestPosterior = doMvnDec(matTrainData,vecTrainTrialType,matTestData,dblLambda);
			%assign output
			matPosteriorProbability(:,~indSelect) = matTestPosterior;
			
			%msg
			if toc(ptrTic) > 5
				ptrTic = tic;
				pause(eps);
				if intVerbose > 0,fprintf('Decoding; now at trial %d/%d [%s]\n',intLeaveOut,intTrials,getTime);end
			end
		end
		
	elseif intTypeCV == 2
		%remove repetition
		if round(intRepNum) ~= intRepNum,error([mfilename ':IncompleteRepetitions'],'Number of repetitions is not an integer');end
		for intRep=1:intRepNum
			%msg
			if toc(ptrTic) > 5
				ptrTic = tic;
				pause(eps);
				if intVerbose > 0,fprintf('Decoding; now at rep %d/%d [%s]\n',intRep,intRepNum,getTime);end
			end
			
			%split trials
			indSelect = true(1,intTrials);
			indSelect(vecRepetition==intRep) = false;
			matTrainData = matData(:,indSelect);
			vecTrainTrialType = vecTrialTypeIdx(indSelect);
			matTestData = matData(:,~indSelect);
			
			%% calculate test probabilities by fitting a multivariate gaussian to the training data
			matTestPosterior = doMvnDec(matTrainData,vecTrainTrialType,matTestData,dblLambda);
			%assign output
			matPosteriorProbability(:,~indSelect) = matTestPosterior;
		end
	else
		error([mfilename ':SyntaxError'],'CV type not recognized');
	end
	
	%% use posterior to calculate classes
	%check if posterior is valid
	if any(isnan(matPosteriorProbability))
		warning([mfilename ':PosteriorNaN'],'Posterior contains NaNs; either your data are corrupt or contain zero-variance predictors. Try increasing the regularization parameter lamdba to improve numerical stability. I have set all nans to 0.');
		matPosteriorProbability(isnan(matPosteriorProbability))=0;
	end
	
	% normal decoding or with prior distro?
	if isempty(vecPriorDistribution)
		%calculate output
		[dummy, vecDecodedIndexCV] = max(matPosteriorProbability,[],1);
	else
		%% loop through trials and assign next most certain trial
		vecDecodedIndexCV = nan(intTrials,1);
		indAssignedTrials = false(intTrials,1);
		matTempProbs = matPosteriorProbability;
		for intTrial=1:intTrials
			%check if we're done
			if sum(vecPriorDistribution==0)==(numel(vecPriorDistribution)-1)
				vecDecodedIndexCV(~indAssignedTrials) = find(vecPriorDistribution>0);
				break;
			end
			
			%remove trials of type that has been chosen max number
			matTempProbs(vecPriorDistribution==0,:) = nan;
			matTempProbs(:,indAssignedTrials) = nan;
			
			%calculate probability of remaining trials and types
			[vecTempProbs,vecTempDecodedIndexCV]=max(matTempProbs,[],1);
			%get 2nd most likely stim per trial
			matDist2 = matTempProbs;
			for intT2=1:intTrials
				matDist2(vecTempDecodedIndexCV(intT2),intT2) = nan;
			end
			[vecTempProbs2,vecTempDecodedIndexCV2]=max(matDist2,[],1);
			
			%use trial with largest difference between most likely and 2nd most likely stimulus
			vecMaxDiff = abs(vecTempProbs2 - vecTempProbs);
			%assign trial
			[dummy,intAssignTrial]=max(vecMaxDiff);
			intAssignType = vecTempDecodedIndexCV(intAssignTrial);
			if vecPriorDistribution(intAssignType) == 0
				intAssignType = vecTempDecodedIndexCV2(intAssignTrial);
			end
			vecDecodedIndexCV(intAssignTrial) = intAssignType;
			indAssignedTrials(intAssignTrial) = true;
			vecPriorDistribution(intAssignType) = vecPriorDistribution(intAssignType) - 1;
			%fprintf('assigned %d to %d; %s\n',intAssignType,intAssignTrial,sprintf('%d ',vecPriorDistribution))
			%pause
		end
	end
	dblPerformanceCV=sum(vecDecodedIndexCV(:) == vecTrialTypeIdx)/length(vecDecodedIndexCV);
	
	
	%error
	if nargout > 3
		if range(vecUniqueTrialTypes) > (2*pi)
			vecUniqueTrialTypes = deg2rad(vecUniqueTrialTypes);
			vecTrialTypes = deg2rad(vecTrialTypes);
		end
		vecDecodedValuesCV = vecUniqueTrialTypes(vecDecodedIndexCV);
		dblMeanErrorRads = mean(abs(circ_dist(vecDecodedValuesCV,vecTrialTypes)));
		dblMeanErrorDegs = rad2deg(dblMeanErrorRads);
	end
	
	%confusion matrix;
	if nargout > 4
		matConfusion = getFillGrid(zeros(intStimTypes),vecDecodedIndexCV,vecTrialTypeIdx,ones(intTrials,1));
		%imagesc(matConfusion,[0 1]);colormap(hot);axis xy;colorbar;
	end
end
function matTestPosterior = doMvnDec(matTrainData,vecTrainTrialType,matTestData,dblLambda)
	%% calculate test probabilities by fitting a multivariate gaussian to the training data
	%get variables
	matSampleData = [matTestData matTrainData];
	[intNeurons,intTrials] = size(matSampleData);
	intStimTypes = length(unique(vecTrainTrialType));
	intTrainTrials = size(matTrainData,2);
	intTestTrials = size(matTestData,2);
	
	%prep data
	matMeans = NaN(intNeurons,intStimTypes);
	for k = 1:intStimTypes
		matMeans(:,k) = mean(matTrainData(:,vecTrainTrialType==k),2);
	end
	
	%center data
	matTrainCentered = matTrainData' - matMeans(:,vecTrainTrialType)';
	
	% QR decomposition
	[Q,R] = qr(matTrainCentered, 0);
	R = R / sqrt(intTrainTrials - intStimTypes); % SigmaHat = R'*R
	s = svd(R);
	logDetSigma = 2*sum(log(s)); % avoid over/underflow
	
	% calculate log probabilities
	D_full = NaN(intTrials, intStimTypes);
	for k = 1:intStimTypes
		A = bsxfun(@minus,matSampleData', matMeans(:,k)') / R;
		D_full(:,k) = log(1/intStimTypes) - .5*(sum(A .* A, 2) + logDetSigma);
	end
	
	if dblLambda > 0 %skip if not required
		%naive Bayes (independent)
		S = std(matTrainCentered) * sqrt((intTrainTrials-1)./(intTrainTrials-intStimTypes));
		D_diag = NaN(intTrials, intStimTypes);
		for k = 1:intStimTypes
			A=bsxfun(@times, bsxfun(@minus,matSampleData',matMeans(:,k)'),1./S);
			D_diag(:,k) = log(1/intStimTypes) - .5*(sum(A .* A, 2) + logDetSigma);
		end
	else
		D_diag = 0;
	end
	
	
	if isinf(dblLambda)%special case to avoid numerial overflow
		%take only D_diag
		D = D_diag;
	else
		%weight probabilities by lambda ratio
		D = (dblLambda*D_diag + D_full) / (1 + dblLambda);
	end
	
	% find highest log probability for each trial
	maxD = max(D, [], 2);
	
	%because of earlier reordering, the first intTestTrials trials are the test set
	% Bayes' rule: first compute p{x,G_j} = p{x|G_j}Pr{G_j} ...
	% (scaled by max(p{x,G_j}) to avoid over/underflow)
	% ... then Pr{G_j|x) = p(x,G_j} / sum(p(x,G_j}) ...
	% (numer and denom are both scaled, so it cancels out)
	
	%likelihoods of test data for each class, scaled to max likelihood
	P = exp(bsxfun(@minus,D(1:intTestTrials,:),maxD(1:intTestTrials)));
	%rescale over P
	sumP = nansum(P,2);
	
	%assign output
	matTestPosterior = bsxfun(@times,P,1./(sumP));
end
