function [dblPerformanceCV,vecDecodedIndexCV,matPosteriorProbabilityCV,dblMeanErrorDegs,matConfusion] = doCrossValidatedDecodingML(matData,vecTrialTypes,intTypeCV,vecPriorDistribution)
	%doCrossValidatedDecodingML Perform cross-validated maximum-likelihood decoding
	%   [dblPerformanceCV,vecDecodedIndexCV,matPosteriorProbabilityCV,dblMeanErrorDegs,matConfusion] = ...
	%		doCrossValidatedDecodingML(matData,vecTrialTypes,intTypeCV,vecPriorDistribution)
	%
	%Inputs:
	%	- matData; [N by T] Neurons by trials response matrix
	%	- vecTrialTypes: [1 by T] Trial type per trial
	%	- intTypeCV: [scalar or vector] Type of cross-validation:
	%		0 = none
	%		1 = leave-one-out (default)
	%		2 = leave-repetition-out
	%		v = vector of training trials (0) and test trials (1)
    %   - vecPriorDistribution: (optional) vector specifying # per trial type
	%
	%Outputs:
	%	- dblPerformanceCV; [scalar] Fraction of correctly decoded trials
	%	- vecDecodedIndexCV: [1 by T] Decoded trial type per trial
	%	- matPosteriorProbabilityCV: [T by U by N] Posterior probability for trials (T) by trial types (U) by neurons (N)
	%	- dblMeanErrorDegs: [scalar] Error in degrees if trial types are radians
	%	- matConfusion: [U by U] Confusion matrix (decoded by real)
	%
	%Version History:
	%1.0 - 16 Jan 2020
	%	Added function description [by Jorrit Montijn]
	%1.1 - 17 Jan 2020
	%	Added train/test set cross-validation [by JM]
    %1.2 - 6 Apr 2020
    %   Added prior distribution enforcement [by JM]
	
	%% prep
	%check which kind of cross-validation
	if ~exist('intTypeCV','var') || isempty(intTypeCV) || (numel(intTypeCV) == 1 && ~(intTypeCV < 3))
		intTypeCV = 1;
	end
	intVerbose = 0;
	
	%prior distribution
	if ~exist('vecPriorDistribution','var')
		vecPriorDistribution = [];
	end
	
	%get number of trials
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
	%remove neurons with range0
	vecAllSd = xstd(matData,2);
	indRem=vecAllSd==0;
	matData(indRem,:)=[];
	
	%get data
	intNeurons = size(matData,1);
	[vecTrialTypeIdx,vecUniqueTrialTypes,vecCounts,cellSelect,vecRepetition] = label2idx(vecTrialTypes);
	intStimTypes = length(vecUniqueTrialTypes);
	intRepNum = min(vecCounts);
	if numel(vecPriorDistribution) ~= intStimTypes || sum(vecPriorDistribution) ~= intTrials
		error([mfilename ':MismatchPriorStimtypes'],'Size of vecPriorDistribution and vecTrialTypes do not match');
	end
	
	%pre-allocate output
	matLikelihood = nan(intNeurons,intStimTypes,2); %mean, sd
	matPosteriorProbabilityCV = nan(intTrials,intStimTypes,intNeurons);
	
	%% cross-validate
	if numel(intTypeCV) > 1
		%train/test set CV
		intTypeCV = intTypeCV(:);
		
		%get train/test data
		indTrainTrials = intTypeCV==min(intTypeCV);
		vecTestTrials = find(intTypeCV==max(intTypeCV));
		vecTrialTypeIdx = vecTrialTypeIdx(vecTestTrials);
		vecAllTrialTypes = vecTrialTypes;
		vecTrialTypes = vecTrialTypes(vecTestTrials);
		
		%change output to only test data
		intTrials = numel(vecTestTrials);
		matPosteriorProbabilityCV = nan(intTrials,intStimTypes,intNeurons);
	
		%build likelihood
		for intStimType=1:intStimTypes
			dblStimTypeValue = vecUniqueTrialTypes(intStimType);
			if intVerbose > 0 && (mod(intStimType,10) == 0 || intTrials > 1000),fprintf('Preparing stimulus %d/%d [%s]\n',intStimType,intStimTypes,getTime);pause(eps);end
			vecMu = xmean(matData(:,vecAllTrialTypes==dblStimTypeValue & indTrainTrials),2); %mean response this trial type per neuron
			vecSD = xstd(matData(:,vecAllTrialTypes==dblStimTypeValue & indTrainTrials),2); %sd response this trial type per neuron
			vecSD(vecSD==0)=vecAllSd(vecSD==0);
			
			%put data in likelihood parameter matrix
			matLikelihood(:,intStimType,1) = vecMu;
			matLikelihood(:,intStimType,2) = vecSD;
			
			%calculate non-CV posterior
			for intTrialIdx = 1:intTrials
				intTrial = vecTestTrials(intTrialIdx);
				matPosteriorProbabilityCV(intTrialIdx,intStimType,:) = normpdf(matData(:,intTrial), matLikelihood(:,intStimType,1),matLikelihood(:,intStimType,2));
			end
		end
		
	elseif intTypeCV == 0
		%no CV
		for intStimType=1:intStimTypes
			dblStimTypeValue = vecUniqueTrialTypes(intStimType);
			if intVerbose > 0 && (mod(intStimType,10) == 0 || intTrials > 1000),fprintf('Preparing stimulus %d/%d [%s]\n',intStimType,intStimTypes,getTime);pause(eps);end
			vecMu = xmean(matData(:,vecTrialTypes==dblStimTypeValue),2); %mean response this trial type per neuron
			vecSD = xstd(matData(:,vecTrialTypes==dblStimTypeValue),2); %sd response this trial type per neuron
			vecSD(vecSD==0)=vecAllSd(vecSD==0);
			
			%put data in likelihood parameter matrix
			matLikelihood(:,intStimType,1) = vecMu;
			matLikelihood(:,intStimType,2) = vecSD;
			
			%calculate non-CV posterior
			for intTrial = 1:intTrials
				matPosteriorProbabilityCV(intTrial,intStimType,:) = normpdf(matData(:,intTrial), matLikelihood(:,intStimType,1),matLikelihood(:,intStimType,2));
			end
		end
	elseif intTypeCV == 1
		%leave one out
		
		%get distances
		for intStimType=1:intStimTypes
			dblStimTypeValue = vecUniqueTrialTypes(intStimType);
			if intVerbose > 0 && (mod(intStimType,10) == 0 || intTrials > 1000),fprintf('Preparing stimulus %d/%d [%s]\n',intStimType,intStimTypes,getTime);pause(eps);end
			vecMu = xmean(matData(:,vecTrialTypes==dblStimTypeValue),2); %mean response this trial type per neuron
			vecSD = xstd(matData(:,vecTrialTypes==dblStimTypeValue),2); %sd response this trial type per neuron
			vecSD(vecSD==0)=vecAllSd(vecSD==0);
			
			%put data in likelihood parameter matrix
			matLikelihood(:,intStimType,1) = vecMu;
			matLikelihood(:,intStimType,2) = vecSD;
			
			%calculate non-CV posterior
			for intTrial = 1:intTrials
				matPosteriorProbabilityCV(intTrial,intStimType,:) = normpdf(matData(:,intTrial), matLikelihood(:,intStimType,1),matLikelihood(:,intStimType,2));
			end
		end
		
		%leave one out
		for intLeaveOut=1:intTrials
			%get info on to-be-left-out trial
			indSelect = ~isnan(vecTrialTypes);
			indSelect(intLeaveOut) = false;
			dblTypeCVTrial = vecTrialTypes(intLeaveOut);
			intTypeCVTrialNumber = find(vecUniqueTrialTypes==dblTypeCVTrial);
			
			%calc CV mean + sd
			vecMuCV = xmean(matData(:,(vecTrialTypes==dblTypeCVTrial)&indSelect),2);
			vecSDCV = xstd(matData(:,(vecTrialTypes==dblTypeCVTrial)&indSelect),2)*(intReps/(intReps - 1));
			vecSDCV(vecSDCV==0)=vecAllSd(vecSDCV==0);
			
			%calc CV parameters
			matLikelihoodCV = matLikelihood;
			matLikelihoodCV(:,intTypeCVTrialNumber,1) = vecMuCV;
			matLikelihoodCV(:,intTypeCVTrialNumber,2) = vecSDCV;
			
			%get posterior probability
			matPosteriorProbabilityCV(intLeaveOut,intTypeCVTrialNumber,:) = normpdf(matData(:,intLeaveOut),matLikelihoodCV(:,intTypeCVTrialNumber,1),matLikelihoodCV(:,intTypeCVTrialNumber,2));
			
			%msg
			if mod(intLeaveOut,100) == 0,pause(eps);end
			if intVerbose > 0 && (mod(intLeaveOut,1000) == 0),fprintf('Decoding; now at trial %d/%d [%s]\n',intLeaveOut,intTrials,getTime);end
		end
	elseif intTypeCV == 2
		%remove repetition
		if round(intRepNum) ~= intRepNum,error([mfilename ':IncompleteRepetitions'],'Number of repetitions is not an integer');end
		for intRep=1:intRepNum
			
			%remove trials
			indTrainTrials = true(1,intTrials);
			indTrainTrials(vecRepetition==intRep) = false;
			matThisTrainData = matData(:,indTrainTrials);
			vecTrainTrialType = vecTrialTypes(indTrainTrials);
			vecTestTrials = find(vecRepetition==intRep);
			intTestTrials = numel(vecTestTrials);
			
			%build likelihood
			matTempLikelihood = nan(intNeurons,intStimTypes,2); %mean, sd
			for intStimType=1:intStimTypes
				dblStimTypeValue = vecUniqueTrialTypes(intStimType);
				vecMu = xmean(matThisTrainData(:,vecTrainTrialType==dblStimTypeValue),2); %mean response this trial type per neuron
				vecSD = xstd(matThisTrainData(:,vecTrainTrialType==dblStimTypeValue),2); %sd response this trial type per neuron
				vecSD(vecSD==0)=vecAllSd(vecSD==0);
			
				%put data in likelihood parameter matrix
				matTempLikelihood(:,intStimType,1) = vecMu;
				matTempLikelihood(:,intStimType,2) = vecSD;
				
				%calculate posterior
				for intTrialIdx = 1:intTestTrials
					intTrial = vecTestTrials(intTrialIdx);
					matPosteriorProbabilityCV(intTrial,intStimType,:) = normpdf(matData(:,intTrial), matTempLikelihood(:,intStimType,1),matTempLikelihood(:,intStimType,2));
				end
			end
			
			%msg
			if mod(intRep,10) == 0,pause(eps);end
			if intVerbose > 0 && (mod(intRep,100) == 0),fprintf('Decoding; now at repetition %d/%d [%s]\n',intRep,intRepNum,getTime);end
		end
	end
	
	%% normal decoding or with prior distro?
	if isempty(vecPriorDistribution)
		%calculate output; use summation in log-domain to avoid numerical errors
		matPosteriorProbabilityCV(matPosteriorProbabilityCV==0)=nan;
		[dummy,vecDecodedIndexCV]=min(nansum(-log(matPosteriorProbabilityCV),3),[],2);
	else
		%loop through trials and assign next most certain trial
		vecDecodedIndexCV = nan(intTrials,1);
		indAssignedTrials = false(intTrials,1);
		for intTrial=1:intTrials
			%remove trials of type that has been chosen max number
			matPosteriorProbabilityCV(:,vecPriorDistribution==0,:) = nan;
			matPosteriorProbabilityCV(indAssignedTrials,:,:) = nan;
			%calculate probability of remaining trials and types
			matP = nansum(-log(matPosteriorProbabilityCV),3);
			[vecMinP,vecTempDecodedIndexCV]=min(matP,[],2);
			matP(bsxfun(@eq,matP,vecMinP)) = nan;
			[vecMin2P,vecTempDecodedIndexCV2]=min(matP,[],2);
			%use trial with largest difference between most likely and 2nd most likely stimulus
			vecMinDiff = vecMin2P - vecMinP;
			%assign trial
			[dummy,intAssignTrial]=max(vecMinDiff);
			intAssignType = vecTempDecodedIndexCV(intAssignTrial);
			if vecPriorDistribution(intAssignType) == 0
				intAssignType = vecTempDecodedIndexCV2(intAssignTrial);
			end
			vecDecodedIndexCV(intAssignTrial) = intAssignType;
			indAssignedTrials(intAssignTrial) = true;
			vecPriorDistribution(intAssignType) = vecPriorDistribution(intAssignType) - 1;
			if vecPriorDistribution(intAssignType) == 0
				matPosteriorProbabilityCV(:,intAssignType) = nan;
			end
		end
	end
	
	%error
	dblPerformanceCV = sum(vecDecodedIndexCV == vecTrialTypeIdx)/length(vecDecodedIndexCV);
	if nargout > 3
		vecDecodedValuesCV = vecUniqueTrialTypes(vecDecodedIndexCV);
		dblMeanErrorRads = mean(abs(circ_dist(vecDecodedValuesCV,vecTrialTypes)));
		dblMeanErrorDegs = rad2ang(dblMeanErrorRads);
	end
	
	
	%confusion matrix;
	if nargout > 4
		matConfusion = getFillGrid(zeros(intStimTypes),vecDecodedIndexCV,vecTrialTypeIdx,ones(intTrials,1))/intRepNum;
		%imagesc(matConfusion,[0 1]);colormap(hot);axis xy;colorbar;
	end
	
	%[dummy,vecDecodedIndexCV]=max(nanprod(matPosteriorProbabilityCV,3),[],2);
	%vecDecodedIndexCV = vecUniqueTrialTypes(vecDecodedIndexCV);
	%dblPerformanceCV = sum(vecDecodedIndexCV == vecTrialTypes)/length(vecDecodedIndexCV);
end

