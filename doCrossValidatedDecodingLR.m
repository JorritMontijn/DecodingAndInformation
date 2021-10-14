function [dblPerformanceCV,vecDecodedIndexCV,matPosteriorProbability,dblMeanErrorDegs,matConfusion,matWeights] = doCrossValidatedDecodingLR(matData,vecTrialTypes,intTypeCV,vecPriorDistribution,dblLambda)
	%doCrossValidatedDecodingLR Logistic regression classifier.
	%[dblPerformanceCV,vecDecodedIndexCV,matPosteriorProbability,dblMeanErrorDegs,matConfusion,matWeights] = ...
	%	doCrossValidatedDecodingLR(matData,vecTrialTypes,intTypeCV,vecPriorDistribution,dblLambda)
	%
	%Inputs:
	% - matData; [n x p]  Matrix of n observations/trials of p predictors/neurons
	% - vecTrialTypes; [n x 1] Trial indexing vector of c classes in n observations/trials
	% - intTypeCV; [int or vec] Integer switch 0-2 or trial repetition vector. 
	%				Val=0, no CV; val=1, leave-one-out CV, val=2 (or
	%				vector), leave-repetition-out. 
	% - vecPriorDistribution: (optional) vector specifying # per trial type
	% - dblLambda; [scalar] Ridge regularization parameter 
	%
	%Outputs:
	% - dblPerformance; [scalar] Fraction of correct classifications
	% - vecDecodedIndexCV; [n x 1] Decoded trial index vector of n observations/trials
	% - matPosteriorProbability; posterior probabilities
	% - dblMeanErrorDegs; [scalar] If vecTrialTypes is in radians, error in degrees
	% - matConfusion; [c x c] Confusion matrix of [(decoded class) x (real class)]
	% - matWeights; weight matrix
	%
	%Version History:
	%2015-xx-xx Created function [by Jorrit Montijn]
	%2019-05-27 Optimized code and added support for trial repetition index
	%			as cross-validation argument [by JM] 
	%2021-10-12 Added prior support and changed argument order to match
	%			other decoding functions [by JM]
	
	%% check which kind of cross-validation
	if ~exist('intTypeCV','var') || isempty(intTypeCV)
		intTypeCV = 2;
	end
	if ~exist('dblLambda','var') || isempty(dblLambda)
		dblLambda = 0;
	end
	if numel(dblLambda) ~= 1
		error([mfilename ':LambdaError'],'Lambda must be scalar');
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
	[vecTrialTypeIdx,vecUniqueTrialTypes,vecCounts,cellSelect,vecRepetition] = label2idx(vecTrialTypes);
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
		%third input is trial repetition index
		vecTrialRepetition = intTypeCV;
		
		%remove repetition
		intRepNum = max(vecTrialRepetition);
		matAggWeights = zeros(intNeurons+1,intStimTypes,intRepNum);
		for intRep=1:intRepNum
			%msg
			if toc(ptrTic) > 5
				ptrTic = tic;
				pause(eps);
				if intVerbose > 0,fprintf('Decoding; now at trial %d/%d [%s]\n',intRep,intRepNum,getTime);end
			end
			
			%remove trials
			indThisRep = vecTrialRepetition==intRep;
			matTrainData = matData(:,~indThisRep);
			vecTrainTrialType = vecTrialTypeIdx(~indThisRep);
			matTestData = matData(:,indThisRep);
			
			%get weights
			[matWeights, vecLLH] = doMnLogReg(matTrainData,vecTrainTrialType,dblLambda);
			matAggWeights(:,:,intRep) = matWeights;
			
			%get performance
			matDataPlusLin = [matTestData; ones(1,size(matTestData,2))];
			matActivation = matWeights'*matDataPlusLin;
			matPosteriorProbability(:,indThisRep) = exp(bsxfun(@minus,matActivation,logsumexp(matActivation,1))); %softmax

		end
		
		
	elseif intTypeCV == 0
		%no CV
		
		%get weights
		[matWeights, vecLLH] = doMnLogReg(matData,vecTrialTypeIdx,dblLambda);
		matAggWeights = matWeights;
		
		%get performance
		matDataPlusLin = [matData; ones(1,size(matData,2))];
		matActivation = matWeights'*matDataPlusLin;
		matPosteriorProbability = exp(bsxfun(@minus,matActivation,logsumexp(matActivation,1))); %softmax

	elseif intTypeCV == 1
		%get prob dens
		error('this does not result in a true cross-validation; still to check what is wrong');
		
		%get weights
		[matWeights, vecLLH] = doMnLogReg(matData,vecTrialTypeIdx,dblLambda);
		matAggWeights = zeros(intNeurons+1,intStimTypes,intTrials);
		
		%get performance
		matDataPlusLin = [matData; ones(1,size(matData,2))];
		matActivation = matWeights'*matDataPlusLin;
		matPosteriorProbability = exp(bsxfun(@minus,matActivation,logsumexp(matActivation,1))); %softmax

		%leave one out
		for intLeaveOut=1:intTrials
			%get info on to-be-left-out trial
			indSelect = ~isnan(vecTrialTypeIdx);
			indSelect(intLeaveOut) = false;
			intTypeCVTrial = vecTrialTypeIdx(intLeaveOut);
			
			%get weights
			[matWeights, vecLLH] = doMnLogReg(matData(:,indSelect),vecTrialTypeIdx(indSelect),dblLambda);
			matAggWeights(:,:,intLeaveOut) = matWeights;
			
			%get performance
			matTestData = matData(:,intTypeCVTrial);
			matDataPlusLin = [matTestData; ones(1,size(matTestData,2))];
			matActivation = matWeights'*matDataPlusLin;
			vecMax = exp(bsxfun(@minus,matActivation,logsumexp(matActivation,1)))';
			matPosteriorProbability(intTypeCVTrial,intLeaveOut) = vecMax(intTypeCVTrial); %softmax
			
			%msg
			if toc(ptrTic) > 5
				ptrTic = tic;
				pause(eps);
				if intVerbose > 0,fprintf('Decoding; now at trial %d/%d [%s]\n',intLeaveOut,intTrials,getTime);end
			end
		end
	elseif intTypeCV == 2
		%remove repetition
		matAggWeights = zeros(intNeurons+1,intStimTypes,intRepNum);
		if round(intRepNum) ~= intRepNum,error([mfilename ':IncompleteRepetitions'],'Number of repetitions is not an integer');end
		for intRep=1:intRepNum
			%msg
			if toc(ptrTic) > 5
				ptrTic = tic;
				pause(eps);
				if intVerbose > 0,fprintf('Decoding; now at rep %d/%d [%s]\n',intRep,intRepNum,getTime);end
			end
			
			%remove trials
			indSelect = true(1,intTrials);
			indSelect(vecRepetition==intRep) = false;
			matTrainData = matData(:,indSelect);
			vecTrainTrialType = vecTrialTypeIdx(indSelect);
			matTestData = matData(:,~indSelect);
			
			%get weights
			[matWeights, vecLLH] = doMnLogReg(matTrainData,vecTrainTrialType,dblLambda);
			matAggWeights(:,:,intRep) = matWeights;
			
			%get performance
			matDataPlusLin = [matTestData; ones(1,size(matTestData,2))];
			matActivation = matWeights'*matDataPlusLin;
			matPosteriorProbability(:,~indSelect) = exp(bsxfun(@minus,matActivation,logsumexp(matActivation,1))); %softmax
		end
	else
		error([mfilename ':SyntaxError'],'CV type not recognized');
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
		vecDecodedValuesCV = deg2rad(vecUniqueTrialTypes(vecDecodedIndexCV));
		dblMeanErrorRads = mean(abs(circ_dist(vecDecodedValuesCV,deg2rad(vecTrialTypes))));
		dblMeanErrorDegs = rad2deg(dblMeanErrorRads);
	end
	
	%confusion matrix;
	if nargout > 4
		matConfusion = getFillGrid(zeros(intStimTypes),vecDecodedIndexCV,vecTrialTypeIdx,ones(intTrials,1));
		%imagesc(matConfusion,[0 1]);colormap(hot);axis xy;colorbar;
	end
	
	%calc aggregate weights
	if nargout > 5
		matWeights = mean(matAggWeights,3);
	end
end

