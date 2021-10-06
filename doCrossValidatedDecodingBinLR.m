function [dblPerformanceBin,vecDecodedIndex,matPosteriorProbability,matWeights,dblMeanErrorDegs,matConfusion] = doCrossValidatedDecodingBinLR(matData,vecTrialTypes,dblLambda)
	%UNTITLED9 Summary of this function goes here
	%   Detailed explanation goes here

	%check which kind of cross-validation
	if nargin < 3 || isempty(dblLambda)
		dblLambda = 0;
	end
	intVerbose = 0;
	
	%get number of trials
	vecTrialTypes = vecTrialTypes(:);
	intTrials = numel(vecTrialTypes);
	error('cross-validation not implemented')
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
	vecUniqueTrialTypes = unique(vecTrialTypes);
	intStimTypes = length(vecUniqueTrialTypes);
	vecTrialTypeIdx = label2idx(vecTrialTypes);
	intReps = intTrials/intStimTypes;
	idx1 = vecTrialTypeIdx==1;
	idx2 = vecTrialTypeIdx==2;
	
	%% multinomial
	tic
	%get weights
	[matWeights, vecLLH] = doMnLogReg(matData,vecTrialTypeIdx,dblLambda);
	%get performance
	matDataPlusLin = [matData; ones(1,size(matData,2))];
	matActivation = matWeights'*matDataPlusLin;
	matPosteriorProbability = exp(bsxfun(@minus,matActivation,logsumexp(matActivation,1))); %softmax
	[dummy, vecDecodedIndex] = max(matPosteriorProbability,[],1);
	dblDprimeMnLogReg = getdprime2(matActivation(1,idx1),matActivation(1,idx2));
	%decoding accuracy
	dblPerformanceMn = sum(vecDecodedIndex(:)==vecTrialTypeIdx(:))/intTrials;
	toc
	%% binary
	tic
	%get logistic regression output
	[vecWeightsLogReg, dblLLH] = doBinLogReg(matData', vecTrialTypeIdx', dblLambda);
	%get performance
	vecActBin = vecWeightsLogReg'*[matData;ones(1,size(matData,2))];
	dblDprimeBinLogReg = getdprime2(vecActBin(idx1),vecActBin(idx2));	
	vecDecodedIndexBin = (vecActBin>0) + 1;
	%decoding accuracy
	dblPerformanceBin = sum(vecDecodedIndexBin(:)==vecTrialTypeIdx(:))/intTrials;
	
	[vecWeights, dblLLH, vecBinaryPrediction, vecPrediction] = doBinLogReg(matData, vecTrialTypeIdx, dblLambda)
	dblPerformanceBin = sum(vecBinaryPrediction(:)==vecTrialTypeIdx(:))/intTrials;
	toc
	%error
	if nargout > 4
		vecDecodedValues = vecUniqueTrialTypes(vecDecodedIndex);
		dblMeanErrorRads = mean(abs(circ_dist(vecDecodedValues,vecTrialTypes)));
		dblMeanErrorDegs = rad2ang(dblMeanErrorRads);
	end
	
	%confusion matrix;
	if nargout > 5
		matConfusion = getFillGrid(zeros(intStimTypes),vecDecodedIndex(:),vecTrialTypeIdx(:),ones(intTrials,1))/intReps;
		%imagesc(matConfusion,[0 1]);colormap(hot);axis xy;colorbar;
	end
	
end

