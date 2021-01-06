function [dblPerformanceCV,vecDecodedIndexCV,matTemplateDistsCV,dblMeanErrorDegs,matConfusion] = doCrossValidatedDecodingTM(matData,vecTrialTypes,intTypeCV)
	%doCrossValidatedDecodingTM Perform cross-validated template matching decoding
	%   [dblPerformanceCV,vecDecodedIndexCV,matPosteriorProbabilityCV,dblMeanErrorDegs,matConfusion] = ...
	%		doCrossValidatedDecodingTM(matData,vecTrialTypes,intTypeCV)
	%
	%Inputs:
	%	- matData; [N by T] Neurons by trials response matrix
	%	- vecTrialTypes: [1 by T] Trial type per trial
	%	- intTypeCV: [scalar or vector] Type of cross-validation:
	%		0 = none
	%		1 = leave-one-out (default)
	%		2 = leave-repetition-out
	%		v = vector of training trials (0) and test trials (1)
	%
	%Outputs:
	%	- dblPerformanceCV; [scalar] Fraction of correctly decoded trials
	%	- vecDecodedIndexCV: [1 by T] Decoded trial type per trial
	%	- matTemplateDistsCV: [U by T] Template distances for trial types (U) by trials (T)
	%	- dblMeanErrorDegs: [scalar] Error in degrees if trial types are radians
	%	- matConfusion: [U by U] Confusion matrix (decoded by real)
	%
	%Version History:
	%1.0 - 28 Jan 2020
	%	Rewrote TM decoder to match syntax of other scripts [by Jorrit Montijn]
	
	%check which kind of cross-validation
	if nargin < 3 || isempty(intTypeCV) || (numel(intTypeCV) == 1 && ~(intTypeCV < 3))
		intTypeCV = 1;
	end
	intVerbose = 0;
	
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
	intNeurons = size(matData,1);
	[vecTrialTypeIdx,vecUniqueTrialTypes,vecCounts,cellSelect,vecRepetition] = label2idx(vecTrialTypes);
	intStimTypes = length(vecUniqueTrialTypes);
	intReps = intTrials/intStimTypes;
	
	%pre-allocate output
	matTemplates = nan(intNeurons,intStimTypes,2); %mean, sd
	matTemplateDistsCV = nan(intStimTypes,intTrials);
		
	%cross-validate
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
		matTemplateDistsCV = nan(intTrials,intStimTypes,intNeurons);
	
		%build templates
		for intStimType=1:intStimTypes
			dblStimTypeValue = vecUniqueTrialTypes(intStimType);
			if intVerbose > 0 && (mod(intStimType,10) == 0 || intTrials > 1000),fprintf('Preparing stimulus %d/%d [%s]\n',intStimType,intStimTypes,getTime);pause(eps);end
			vecMu = xmean(matData(:,vecAllTrialTypes==dblStimTypeValue & indTrainTrials),2); %mean response this trial type per neuron
			vecSD = xstd(matData(:,vecAllTrialTypes==dblStimTypeValue & indTrainTrials),2); %sd response this trial type per neuron
			
			%put data in likelihood parameter matrix
			matTemplates(:,intStimType,1) = vecMu;
			matTemplates(:,intStimType,2) = vecSD;
		end
		
		%calculate dists
		for intTrialIdx = 1:intTrials
			intTrial = vecTestTrials(intTrialIdx);
			%get this response
			vecR = matData(:,intTrial);
			
			%calculate distances
			matDists = abs(bsxfun(@rdivide,bsxfun(@minus,vecR,matTemplates(:,:,1)),matTemplates(:,:,2)));
			matDists(isinf(matDists)) = nan;
			
			%put in output matrix
			matTemplateDistsCV(:,intTrialIdx) = nansum(matDists,1)./sum(~isnan(matDists),1);
		end
	elseif intTypeCV == 0
		%no CV
		for intStimType=1:intStimTypes
			dblStimTypeValue = vecUniqueTrialTypes(intStimType);
			if intVerbose > 0 && (mod(intStimType,10) == 0 || intTrials > 1000),fprintf('Preparing stimulus %d/%d [%s]\n',intStimType,intStimTypes,getTime);pause(eps);end
			vecMu = xmean(matData(:,vecTrialTypes==dblStimTypeValue),2); %mean response this trial type per neuron
			vecSD = xstd(matData(:,vecTrialTypes==dblStimTypeValue),2); %sd response this trial type per neuron
			
			%put data in likelihood parameter matrix
			matTemplates(:,intStimType,1) = vecMu;
			matTemplates(:,intStimType,2) = vecSD;
		end
		
		%calculate non-CV dists
		for intTrial = 1:intTrials
			%get this response
			vecR = matData(:,intTrial);
			
			%calculate distances
			matDists = abs(bsxfun(@rdivide,bsxfun(@minus,vecR,matTemplates(:,:,1)),matTemplates(:,:,2)));
			matDists(isinf(matDists)) = nan;
			
			%put in output matrix
			matTemplateDistsCV(:,intTrial) = nansum(matDists,1)./sum(~isnan(matDists),1);
		end
		
	elseif intTypeCV == 1
		%leave one out
		
		%get templates
		matTemplates = zeros(intNeurons,intStimTypes,2);
		for intStimType=1:intStimTypes
			dblStimTypeValue = vecUniqueTrialTypes(intStimType);
			if intVerbose > 0 && (mod(intStimType,10) == 0 || intTrials > 1000),fprintf('Preparing stimulus %d/%d [%s]\n',intStimType,intStimTypes,getTime);pause(eps);end
			vecMu = xmean(matData(:,vecTrialTypes==dblStimTypeValue),2);
			vecSD = xstd(matData(:,vecTrialTypes==dblStimTypeValue),2);
			
			matTemplates(:,intStimType,1) = vecMu;
			matTemplates(:,intStimType,2) = vecSD;
		end
		
		%cross-validate
		for intLeaveOut=1:intTrials
			%get info on to-be-left-out trial
			indSelect = ~isnan(vecTrialTypes);
			indSelect(intLeaveOut) = false;
			dblTypeCVTrial = vecTrialTypes(intLeaveOut);
			intTypeCVTrialNumber = find(vecUniqueTrialTypes==dblTypeCVTrial);
			
			%calc CV mean & SD
			vecMuCV = mean(matData(:,vecTrialTypes==dblTypeCVTrial&indSelect),2);
			vecSDCV = std(matData(:,vecTrialTypes==dblTypeCVTrial&indSelect),[],2);
			
			%calc CV templates
			matTemplatesCV = matTemplates;
			matTemplatesCV(:,intTypeCVTrialNumber,1) = vecMuCV;
			matTemplatesCV(:,intTypeCVTrialNumber,2) = vecSDCV;
			
			%get this response
			vecR = matData(:,intLeaveOut);
			
			%calculate distances
			matDists = abs(bsxfun(@rdivide,bsxfun(@minus,vecR,matTemplatesCV(:,:,1)),matTemplatesCV(:,:,2)));
			matDists(isinf(matDists)) = nan;
			
			%put in output matrix
			matTemplateDistsCV(:,intLeaveOut) = nansum(matDists,1)./sum(~isnan(matDists),1);
			
			%msg
			if mod(intLeaveOut,100) == 0,pause(eps);end
			if mod(intLeaveOut,1000) == 0,fprintf('Decoding; now at trial %d/%d [%s]\n',intLeaveOut,intTrials,getTime);end
		end
	
	elseif intTypeCV == 2
		%remove repetition
		intRepNum = intTrials/intStimTypes;
		if round(intRepNum) ~= intRepNum,error([mfilename ':IncompleteRepetitions'],'Number of repetitions is not an integer');end
		for intRep=1:intRepNum
			
			%remove trials
			indSelect = true(1,intTrials);
			indSelect(vecRepetition==intRep) = false;
			matThisTrainData = matData(:,indSelect);
			vecThisTrainTrialType = vecTrialTypes(indSelect);
			
			%get templates
			matTemplatesCV = zeros(intNeurons,intStimTypes,2);
			for intStimType=1:intStimTypes
				dblStimTypeValue = vecUniqueTrialTypes(intStimType);
				if intVerbose > 0 && (mod(intStimType,10) == 0 || intTrials > 1000),fprintf('Preparing stimulus %d/%d [%s]\n',intStimType,intStimTypes,getTime);pause(eps);end
				vecMu = xmean(matThisTrainData(:,vecThisTrainTrialType==dblStimTypeValue),2);
				vecSD = xstd(matThisTrainData(:,vecThisTrainTrialType==dblStimTypeValue),2);
				
				matTemplatesCV(:,intStimType,1) = vecMu;
				matTemplatesCV(:,intStimType,2) = vecSD;
			end
			
			
			%get posterior for trials in repetition
			for intTrialCV=vecThisRepTrials
				%get this response
				vecR = matData(:,intTrialCV);
				
				%calculate distances
				matDists = abs(bsxfun(@rdivide,bsxfun(@minus,vecR,matTemplatesCV(:,:,1)),matTemplatesCV(:,:,2)));
				matDists(isinf(matDists)) = nan;
				
				%put in output matrix
				matTemplateDistsCV(:,intTrialCV) = nansum(matDists,1)./sum(~isnan(matDists),1);
			end
			
			%msg
			if mod(intRep,10) == 0,pause(eps);end
			if intVerbose > 0 && (mod(intRep,100) == 0),fprintf('Decoding; now at repetition %d/%d [%s]\n',intRep,intRepNum,getTime);end
		end
	end
	
	%calculate output
	[dummy,vecDecodedIndexCV]=min(matTemplateDistsCV,[],1);	
	dblPerformanceCV=sum(vecDecodedIndexCV(:) == vecTrialTypeIdx)/length(vecDecodedIndexCV);
	
	%error
	if nargout > 3
		vecDecodedValuesCV = vecUniqueTrialTypes(vecDecodedIndexCV);
		dblMeanErrorRads = mean(abs(circ_dist(vecDecodedValuesCV,vecTrialTypes)));
		dblMeanErrorDegs = rad2ang(dblMeanErrorRads);
	end
	
	
	%confusion matrix;
	if nargout > 4
		matConfusion = getFillGrid(zeros(intStimTypes),vecDecodedIndexCV,vecTrialTypeIdx,ones(intTrials,1))/intReps;
		%imagesc(matConfusion,[0 1]);colormap(hot);axis xy;colorbar;
	end
	
	%[dummy,vecDecodedIndexCV]=max(nanprod(matPosteriorProbabilityCV,3),[],2);
	%vecDecodedIndexCV = vecUniqueTrialTypes(vecDecodedIndexCV);
	%dblPerformanceCV = sum(vecDecodedIndexCV == vecTrialTypes)/length(vecDecodedIndexCV);
end

