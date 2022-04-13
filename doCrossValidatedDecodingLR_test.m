function [dblPerformanceCV,vecDecodedIndexCV,matPosteriorProbability,dblMeanErrorDegs,matConfusion,matWeights,matAggActivation] = doCrossValidatedDecodingLR(matData,vecTrialTypes,intTypeCV,vecPriorDistribution,dblLambda)
	%doCrossValidatedDecodingLR Logistic regression classifier.
	%[dblPerformanceCV,vecDecodedIndexCV,matPosteriorProbability,dblMeanErrorDegs,matConfusion,matWeights,matAggActivation] = ...
	%	doCrossValidatedDecodingLR(matData,vecTrialTypes,intTypeCV,vecPriorDistribution,dblLambda)
	%
	%CV GLM classifier
	
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
	[vecTrialTypeIdx,vecUniqueTrialTypes,vecCounts,cellSelect,vecRepetition] = val2idx(vecTrialTypes);
	intStimTypes = length(vecUniqueTrialTypes);
	intRepNum = min(vecCounts);
	if ~isempty(vecPriorDistribution) && (numel(vecPriorDistribution) ~= intStimTypes || sum(vecPriorDistribution) ~= intTrials)
		error([mfilename ':MismatchPriorStimtypes'],'Size of vecPriorDistribution and vecTrialTypes do not match');
	end
	
	%pre-allocate output
	matPosteriorProbability = zeros(intStimTypes,intTrials);
	matAggActivation = zeros(intStimTypes,intTrials);
	vecDecodedIndexCV = nan(1,intTrials);
	
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
		%remove repetition
		%matAggWeights = zeros(intNeurons+1,intStimTypes-1,intRepNum);
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
			vecTestTrialType = vecTrialTypeIdx(~indSelect);
			
			%glm
			%{
			mdl = fitglm(matTrainData',vecTrainTrialType);
			ypred = predict(mdl,matTestData');
			yDist = ypred - vecTestTrialType';
			[vecMinDist,vecDecodedIdx] = min(abs(yDist),[],2);
			vecDecodedIndexCV(~indSelect) = vecDecodedIdx;
			%}
			%lda
			%%{
			[class,err,POSTERIOR,logp,coeff] = classify(matTestData',matTrainData',vecTrainTrialType,'linear');
			vecDecodedIndexCV(~indSelect) = class;
			matPosteriorProbability(:,~indSelect) = POSTERIOR;
			
			%}
			
			%nbKD = fitctree(matTrainData', vecTrainTrialType);
			%nbKD = fitcnb(matTrainData', vecTrainTrialType, 'DistributionNames','kernel', 'Kernel','box');
			%vecDecodedIndexCV(~indSelect) = predict(nbKD, matTestData');
		end
	else
		error([mfilename ':SyntaxError'],'CV type not recognized');
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

