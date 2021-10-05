function [dblPerformance,vecDecodedIndexCV,matMahalDistsCV,dblMeanErrorDegs,matConfusion] = doCrossValidatedDecodingMD(matData,vecTrialTypes,intTypeCV)
	%doCrossValidatedDecodingMD Multi-dimensional Mahalanobis-Distance classifier.
	%[dblPerformance,vecDecodedIndexCV,matMahalDistsCV,dblMeanErrorDegs,matConfusion] = ...
	%	doCrossValidatedDecodingMD(matData,vecTrialTypes,intTypeCV)
	%
	%Inputs:
	% - matData; [n x p]  Matrix of n observations/trials of p predictors/neurons
	% - vecTrialTypes; [n x 1] Trial indexing vector of c classes in n observations/trials
	% - intTypeCV; [int or vec] Integer switch 0-2 or trial repetition vector. 
	%				Val=0, no CV; val=1, leave-one-out CV, val=2 (or
	%				vector), leave-repetition-out. 
	%
	%Outputs:
	% - dblPerformance; [scalar] Fraction of correct classifications
	% - vecDecodedIndexCV; [n x 1] Decoded trial index vector of n observations/trials
	% - matMahalDistsCV; [n x p] Mahalanobis distance to class mean
	% - dblMeanErrorDegs; [scalar] If vecTrialTypes is in radians, error in degrees
	% - matConfusion; [c x c] Confusion matrix of [(decoded class) x (real class)]
	%
	%Version History:
	%2015-xx-xx Created function [by Jorrit Montijn]
	%2019-05-27 Optimized code and added support for trial repetition index
	%			as cross-validation argument [by JM] 
	
	%% check which kind of cross-validation
	if nargin < 3 || isempty(intTypeCV)
		intTypeCV = 1;
	end
	%% prepare
	intVerbose = 1;
	boolUseMahalInvCov = true;
	
	%get number of trials
	vecTrialTypes = vecTrialTypes(:);
	intTrials = numel(vecTrialTypes);
	
	%check if matData is [trial x neuron] or [neuron x trial]
	if size(matData,1) == intTrials && size(matData,2) == intTrials
		%number of neurons and trials is the same
		warning([mfilename ':SameNeuronsTrials'],'Number of neurons and trials is identical; please double check the proper orientation of [intNeurons x intTrials]');
	elseif size(matData,1) == intTrials
		%size is correct
	elseif size(matData,2) == intTrials
		%rotate
		matData = matData';
	else
		error([mfilename ':SameNeuronsTrials'],'Size of matData and vecTrialTypes do not match');
	end
	intNeurons = size(matData,2);
	[vecTrialTypeIdx,vecUniqueTrialTypes,vecCounts,cellSelect,vecRepetition] = label2idx(vecTrialTypes);
	intStimTypes = length(vecUniqueTrialTypes);
	intRepNum = min(vecCounts);
	
	%pre-allocate output
	matMahalDistsCV = zeros(intTrials,intStimTypes);
	
	%% cross-validate
	if numel(intTypeCV) == intTrials
		%third input is trial repetition index
		vecTrialRepetition = intTypeCV;
		
		%remove repetition
		intRepNum = max(vecTrialRepetition);
		for intRep=1:intRepNum
			%msg
			if mod(intRep,10) == 0,pause(eps);end
			if mod(intRep,10) == 0 && intVerbose > 0,fprintf('Decoding; now at rep %d/%d [%s]\n',intRep,intRepNum,getTime);end

			%remove trials
			indThisRep = vecTrialRepetition==intRep;
			matTrainData = matData(~indThisRep,:);
			vecTrainTrialType = vecTrialTypeIdx(~indThisRep);
			matTestData = matData(indThisRep,:);
			
			%recalculate covariance matrix
			for intStimType=1:intStimTypes
				%get distances
				vecMuTrain = mean(matTrainData(vecTrainTrialType==intStimType,:),1);
				matCovarTrain = cov(matTrainData(vecTrainTrialType==intStimType,:));
				
				if boolUseMahalInvCov
					matCovarInvTrain = inv(matCovarTrain);
					vecMahal=getMahal(matTestData,vecMuTrain,matCovarInvTrain);
				else
					vecMahal=getMahal2(matTestData,vecMuTrain,matCovarTrain);
				end
				matMahalDistsCV(indThisRep,intStimType) = vecMahal;
			end
		end
		
		
	elseif intTypeCV == 0
		%no CV
		for intStimType=1:intStimTypes
			vecMu = mean(matData(vecTrialTypeIdx==intStimType,:),1);
			matCovar = cov(matData(vecTrialTypeIdx==intStimType,:));
			if boolUseMahalInvCov
				matCovarInv = inv(matCovar);
				vecMahal=getMahal(matData,vecMu,matCovarInv);
			else
				vecMahal=getMahal2(matData,vecMu,matCovar);
			end
			matMahalDistsCV(:,intStimType) = vecMahal;
		end
	elseif intTypeCV == 1
		%get distances
		for intStimType=1:intStimTypes
			if mod(intStimType,10) == 0 && intVerbose > 0,fprintf('Preparing stimulus %d/%d [%s]\n',intStimType,intStimTypes,getTime);pause(eps);end
			vecMu = mean(matData(vecTrialTypeIdx==intStimType,:),1);
			matCovar = cov(matData(vecTrialTypeIdx==intStimType,:));
			if boolUseMahalInvCov
				matCovarInv = inv(matCovar);
				vecMahal=getMahal(matData,vecMu,matCovarInv);
			else
				vecMahal=getMahal2(matData,vecMu,matCovar);
			end
			matMahalDistsCV(:,intStimType) = vecMahal;
		end
		
		%leave one out
		for intLeaveOut=1:intTrials
			%get info on to-be-left-out trial
			indSelect = ~isnan(vecTrialTypeIdx);
			indSelect(intLeaveOut) = false;
			intTypeCVTrial = vecTrialTypeIdx(intLeaveOut);
			
			%calc CV mean
			vecMuTrain = mean(matData((vecTrialTypeIdx==intTypeCVTrial)&indSelect,:),1);
			
			%calc CV covar
			matCovarTrain = cov(matData((vecTrialTypeIdx==intTypeCVTrial)&indSelect,:));
			%get CV mahal dists
			if boolUseMahalInvCov
				matCovarInvTrain = inv(matCovarTrain);
				matMahalDistsCV(intLeaveOut,intTypeCVTrial) = getMahal(matData(intLeaveOut,:),vecMuTrain,matCovarInvTrain);
			else
				matMahalDistsCV(intLeaveOut,intTypeCVTrial) = getMahal2(matData(intLeaveOut,:),vecMuTrain,matCovarTrain);
			end
				
			%msg
			if mod(intLeaveOut,100) == 0,pause(eps);end
			if mod(intLeaveOut,1000) == 0 && intVerbose > 0,fprintf('Decoding; now at trial %d/%d [%s]\n',intLeaveOut,intTrials,getTime);end
		end
	elseif intTypeCV == 2
		%remove repetition
		if round(intRepNum) ~= intRepNum,error([mfilename ':IncompleteRepetitions'],'Number of repetitions is not an integer');end
		intTrial = 0;
		for intRep=1:intRepNum
			
			%remove trials
			indSelect = true(1,intTrials);
			indSelect(vecRepetition==intRep) = false;
			matTrainData = matData(indSelect,:);
			vecTrainTrialType = vecTrialTypeIdx(indSelect);
			
			%recalculate covariance matrix
			for intStimType=1:intStimTypes
				%get distances
				vecMuTrain = mean(matTrainData(vecTrainTrialType==intStimType,:),1);
				matCovarTrain = cov(matTrainData(vecTrainTrialType==intStimType,:));
				
				if boolUseMahalInvCov
					matCovarInvTrain = inv(matCovarTrain);
					vecMahal=getMahal(matData(~indSelect,:),vecMuTrain,matCovarInvTrain);
				else
					vecMahal=getMahal2(matData(~indSelect,:),vecMuTrain,matCovarTrain);
				end
				matMahalDistsCV(~indSelect,intStimType) = vecMahal;
				
				%msg
				intTrial = intTrial + 1;
				if mod(intTrial,100) == 0,pause(eps);end
				if mod(intTrial,1000) == 0 && intVerbose > 0,fprintf('Decoding; now at trial %d/%d [%s]\n',intTrial,intTrials,getTime);end
			end
		end
	else
		error([mfilename ':SyntaxError'],'CV type not recognized');
	end
	
	%output
	[dummy,vecDecodedIndexCV]=min(matMahalDistsCV,[],2);
	dblPerformance=sum(vecDecodedIndexCV==vecTrialTypeIdx)/intTrials;
	
	%error
	if nargout > 3
		vecDecodedValuesCV = vecUniqueTrialTypes(vecDecodedIndexCV);
		dblMeanErrorRads = mean(abs(circ_dist(vecDecodedValuesCV,vecTrialTypes)));
		dblMeanErrorDegs = rad2ang(dblMeanErrorRads);
	end
	
	
	%confusion matrix;
	if nargout > 4
		matConfusion = getFillGrid(zeros(intStimTypes),vecDecodedIndexCV,vecTrialTypeIdx,ones(intTrials,1));
		%imagesc(matConfusion,[0 1]);colormap(hot);axis xy;colorbar;
	end
end

