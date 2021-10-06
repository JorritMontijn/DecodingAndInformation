function [dblPerformance,vecDecodedIndexCV,matPosteriorProbability,matWeights,dblMeanErrorDegs,matConfusion] = doCrossValidatedDecodingLR(matData,vecTrialTypes,intTypeCV,dblLambda)
	%doCrossValidatedDecodingLR Logistic regression classifier.
	%[dblPerformance,vecDecodedIndexCV,matPosteriorProbability,matWeights,dblMeanErrorDegs,matConfusion] = ...
	%	doCrossValidatedDecodingLR(matData,vecTrialTypes,intTypeCV,dblLambda)
	%
	%Inputs:
	% - matData; [n x p]  Matrix of n observations/trials of p predictors/neurons
	% - vecTrialTypes; [n x 1] Trial indexing vector of c classes in n observations/trials
	% - intTypeCV; [int or vec] Integer switch 0-2 or trial repetition vector. 
	%				Val=0, no CV; val=1, leave-one-out CV, val=2 (or
	%				vector), leave-repetition-out. 
	% - dblLambda; [scalar] Ridge regularization parameter 
	%
	%Outputs:
	% - dblPerformance; [scalar] Fraction of correct classifications
	% - vecDecodedIndexCV; [n x 1] Decoded trial index vector of n observations/trials
	% - matPosteriorProbability; posterior probabilities
	% - matWeights; weight matrix
	% - dblMeanErrorDegs; [scalar] If vecTrialTypes is in radians, error in degrees
	% - matConfusion; [c x c] Confusion matrix of [(decoded class) x (real class)]
	%
	%Version History:
	%2015-xx-xx Created function [by Jorrit Montijn]
	%2019-05-27 Optimized code and added support for trial repetition index
	%			as cross-validation argument [by JM] 
	
	%% check which kind of cross-validation
	if nargin < 3 || isempty(intTypeCV)
		intTypeCV = 2;
	end
	if nargin < 4 || isempty(dblLambda)
		dblLambda = 0;
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
	
	%pre-allocate output
	matPosteriorProbability = zeros(intTrials,intStimTypes);
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
			matPosteriorProbability(indThisRep,:) = exp(bsxfun(@minus,matActivation,logsumexp(matActivation,1)))'; %softmax

		end
		
		
	elseif intTypeCV == 0
		%no CV
		
		%get weights
		[matWeights, vecLLH] = doMnLogReg(matData,vecTrialTypeIdx,dblLambda);
		matAggWeights = matWeights;
		
		%get performance
		matDataPlusLin = [matData; ones(1,size(matData,2))];
		matActivation = matWeights'*matDataPlusLin;
		matPosteriorProbability = exp(bsxfun(@minus,matActivation,logsumexp(matActivation,1)))'; %softmax

	elseif intTypeCV == 1
		%get prob dens
		error('this does not result in a true cross-validation; still to check what is wrong');
		
		%get weights
		[matWeights, vecLLH] = doMnLogReg(matData,vecTrialTypeIdx,dblLambda);
		matAggWeights = zeros(intNeurons+1,intStimTypes,intTrials);
		
		%get performance
		matDataPlusLin = [matData; ones(1,size(matData,2))];
		matActivation = matWeights'*matDataPlusLin;
		matPosteriorProbability = exp(bsxfun(@minus,matActivation,logsumexp(matActivation,1)))'; %softmax

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
			matPosteriorProbability(intLeaveOut,intTypeCVTrial) = vecMax(intTypeCVTrial); %softmax
			
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
			matPosteriorProbability(~indSelect,:) = exp(bsxfun(@minus,matActivation,logsumexp(matActivation,1)))'; %softmax
		end
	else
		error([mfilename ':SyntaxError'],'CV type not recognized');
	end
	
	%output
	[dummy, vecDecodedIndexCV] = max(matPosteriorProbability,[],2);
	dblPerformance=sum(vecDecodedIndexCV==vecTrialTypeIdx)/intTrials;
	matWeights = mean(matAggWeights,3);
	
	%error
	if nargout > 4
		vecDecodedValuesCV = deg2rad(vecUniqueTrialTypes(vecDecodedIndexCV));
		dblMeanErrorRads = mean(abs(circ_dist(vecDecodedValuesCV,deg2rad(vecTrialTypes))));
		dblMeanErrorDegs = rad2deg(dblMeanErrorRads);
	end
	
	%confusion matrix;
	if nargout > 5
		matConfusion = getFillGrid(zeros(intStimTypes),vecDecodedIndexCV,vecTrialTypeIdx,ones(intTrials,1));
		%imagesc(matConfusion,[0 1]);colormap(hot);axis xy;colorbar;
	end
end

