function [matMahalDists,matMahalDistsShuffled,vecGroupSizes] = getMahalCV(matData,vecTrialStimType,sParams)
	%getMahalCV Does Mahal Anal. Syntax:
	%   [matMahalDists,matMahalDistsShuffled] = getMahalCV(matData,vecTrialStimType,sParams)
	%Input fields:
	%	intFoldK		(10)
	%	vecUseStimTypes ([1 2])
	%	vecGroupSizes	(trial dependent)
	%	boolVerbose		(true)
	%	dblDiffTheta	(1)
	%	intIters		(10)
	%Output:
	%	matMahalDists	[(real class) x (target class) x (trial) x (iteration) x (group size)]
	%	matMahalDistsShuffled
	%
	%	By Jorrit Montijn, 07-07-17 (dd-mm-yy; Universite de Geneve)
	
	%% check input
	if ~exist('sParams','var'),sParams=struct;end
	if isfield(sParams,'intFoldK'),intFoldK=sParams.intFoldK;else intFoldK = 10;end
	if isfield(sParams,'vecUseStimTypes'),vecUseStimTypes=sParams.vecUseStimTypes;
	else
		[x,y,vecUseStimTypes] = find(sort(unique(vecTrialStimType),'ascend'),2,'first');
		%vecUseStimTypes = find(vecTrialContrasts==100); %TEMP
	end
	if isfield(sParams,'vecGroupSizes'),vecGroupSizes=sParams.vecGroupSizes;
	else
		vecGroupSizes = [2.^[0:20]];
		vecGroupSizes(vecGroupSizes>min([size(matData,1) numel(vecTrialStimType)/numel(unique(vecTrialStimType))]))=[];
		vecGroupSizes((end-1):end) = [];
		%vecGroupSizes = 2.^(0:8); %TEMP
	end
	if isfield(sParams,'intIters'),intIters=sParams.intIters;else intIters = 10;end
	if isfield(sParams,'boolVerbose'),boolVerbose=sParams.boolVerbose;else boolVerbose = true;end
	if isfield(sParams,'dblDiffTheta'),dblDiffTheta=sParams.dblDiffTheta;else dblDiffTheta = 1;end
	if nargout > 1,boolDoShuffled = true;end
	
	%% general data (not class-specific)
	intGroups = numel(vecGroupSizes);
	intNeurons = size(matData,1);
	
	%% prep data (12-class-specific)
	indKeepTrials = ismember(vecTrialStimType,vecUseStimTypes);
	vecClasses12 = label2idx(vecTrialStimType(indKeepTrials));
	matData12 =  matData(:,indKeepTrials);
	
	%% build folds
	intTrialsPerFold = floor(min([sum(vecClasses12==1) sum(vecClasses12==2)])/intFoldK); %take lowest number of repetitions for the two classes, and round down when dividing in equal K-folds
	cellFolds1 = cell(1,intFoldK);
	cellFolds2 = cell(1,intFoldK);
	cellFoldsShuffled1 = cell(1,intFoldK);
	cellFoldsShuffled2 = cell(1,intFoldK);
	vecClasses1 = find(vecClasses12==1);
	vecClasses2 = find(vecClasses12==2);
	for intFold=1:intFoldK
		vecTrials = (intTrialsPerFold*(intFold-1)+1):(intTrialsPerFold*intFold);
		cellFolds1{intFold} = matData12(:,vecClasses1(vecTrials));
		cellFolds2{intFold} = matData12(:,vecClasses2(vecTrials));
		
		matFoldShuff1 = nan(intNeurons,intTrialsPerFold);
		matFoldShuff2 = nan(intNeurons,intTrialsPerFold);
		for intThisNeuron=1:intNeurons
			vecData1 = matData12(intThisNeuron,vecClasses1(vecTrials));
			matFoldShuff1(intThisNeuron,:) = vecData1(randperm(intTrialsPerFold));
			%matFoldShuff1(intThisNeuron,:) = circshift(matData12(intThisNeuron,vecClasses1(vecTrials)),intThisNeuron-1,2);
			vecData2 = matData12(intThisNeuron,vecClasses2(vecTrials));
			matFoldShuff2(intThisNeuron,:) = vecData2(randperm(intTrialsPerFold));
			%matFoldShuff2(intThisNeuron,:) = circshift(matData12(intThisNeuron,vecClasses2(vecTrials)),intThisNeuron,2);
		end
		cellFoldsShuffled1{intFold} = matFoldShuff1;
		cellFoldsShuffled2{intFold} = matFoldShuff2;
	end
	
	%% pre-allocate output
	matMahalDists = []; %[(real class) x (target class) x (trial) x (iteration) x (group size)]
	matMahalDistsShuffled = []; %[(real class) x (target class) x (trial) x (iteration) x (group size)]
	
	%% run
	for intGroupSizeIdx=1:intGroups
		intGroupSize = vecGroupSizes(intGroupSizeIdx);
		if intGroupSize > 100
			intIters = min([20 intIters]);
		end
		if boolVerbose,fprintf('Doing Mahal Anal; now at group size %d (%d/%d) [%s]\n',intGroupSize,intGroupSizeIdx,intGroups,getTime);end
		
		%clean iters
		matMahalDistsIters = [];
		matMahalDistsItersShuffled = [];
		for intIter=1:intIters
			if intNeurons == intGroupSize
				vecNeurons = 1:intNeurons;
			else
				vecNeurons = randperm(intNeurons,intGroupSize);
			end
			if boolVerbose,fprintf('   Iteration %d/%d [%s]\n',intIter,intIters,getTime);end
			
			%clean folds
			matMahalDistsFolds = [];
			matMahalDistsFoldsShuffled = [];
			for intFold=1:intFoldK
				%% get correction factors
				intTrials12 = (intTrialsPerFold)*(intFoldK-1); %check if this one
				dblSubFac =(2*intGroupSize)/(intTrials12*(dblDiffTheta.^2));
				dblProdFacRaw = ((2*intTrials12-intGroupSize-3)/(2*intTrials12-2));
				
				%% real
				%get training & test set
				indFolds = true(1,intFoldK);
				indFolds(intFold) = false;
				matTrain1 = cell2mat(cellFolds1(indFolds));
				matTrain1 = matTrain1(vecNeurons,:);
				matTest1 = cell2mat(cellFolds1(~indFolds));
				matTest1 = matTest1(vecNeurons,:);
				matTrain2 = cell2mat(cellFolds2(indFolds));
				matTrain2 = matTrain2(vecNeurons,:);
				matTest2 = cell2mat(cellFolds2(~indFolds));
				matTest2 = matTest2(vecNeurons,:);
				
				%no CV
				intTestT1 = size(matTest1,2);
				vecMuTrain1 = xmean(matTrain1,2);
				matCovarTrain1 = cov(matTrain1');
				intTestT2 = size(matTest2,2);
				vecMuTrain2 = xmean(matTrain2,2);
				matCovarTrain2 = cov(matTrain2');
				
				%mahal
				matMahal=nan([2 2 size(matTest1,2)]);
				for intTrial = 1:intTestT1 %is actually class 1
					vecXY1 = (matTest1(:,intTrial)-vecMuTrain1);
					matMahal(1,1,intTrial) = vecXY1' * (matCovarTrain1 \ vecXY1);
					vecXY2 = (matTest1(:,intTrial)-vecMuTrain2);
					matMahal(1,2,intTrial) = vecXY2' * (matCovarTrain2 \ vecXY2);
				end
				for intTrial = 1:intTestT2 %is actually class 2
					vecXY1 = (matTest2(:,intTrial)-vecMuTrain1);
					matMahal(2,1,intTrial) = vecXY1' * (matCovarTrain1 \ vecXY1);
					vecXY2 = (matTest2(:,intTrial)-vecMuTrain2);
					matMahal(2,2,intTrial) = vecXY2' * (matCovarTrain2 \ vecXY2);
				end
				matMahal(matMahal==0)=nan;
				[dummy,vecDecodedIndexMD]=min(matMahal,[],3);
				
				%decoding accuracy
				matMahalDistsFolds = cat(3,matMahalDistsFolds,matMahal);
				
				%% shuffled
				if boolDoShuffled
					%get training & test set
					indFolds = true(1,intFoldK);
					indFolds(intFold) = false;
					matTrain1 = cell2mat(cellFoldsShuffled1(indFolds));
					matTrain1 = matTrain1(vecNeurons,:);
					matTest1 = cell2mat(cellFoldsShuffled1(~indFolds));
					matTest1 = matTest1(vecNeurons,:);
					matTrain2 = cell2mat(cellFoldsShuffled2(indFolds));
					matTrain2 = matTrain2(vecNeurons,:);
					matTest2 = cell2mat(cellFoldsShuffled2(~indFolds));
					matTest2 = matTest2(vecNeurons,:);
					
					%no CV
					intTestT1 = size(matTest1,2);
					vecMuTrain1 = xmean(matTrain1,2);
					matCovarTrain1 = cov(matTrain1');
					intTestT2 = size(matTest2,2);
					vecMuTrain2 = xmean(matTrain2,2);
					matCovarTrain2 = cov(matTrain2');
					
					%mahal
					matMahalShuffled=nan([2 2 size(matTest1,2)]);
					for intTrial = 1:intTestT1 %is actually class 1
						vecXY1 = (matTest1(:,intTrial)-vecMuTrain1);
						matMahalShuffled(1,1,intTrial) = vecXY1' * (matCovarTrain1 \ vecXY1);
						vecXY2 = (matTest1(:,intTrial)-vecMuTrain2);
						matMahalShuffled(1,2,intTrial) = vecXY2' * (matCovarTrain2 \ vecXY2);
					end
					for intTrial = 1:intTestT2 %is actually class 2
						vecXY1 = (matTest2(:,intTrial)-vecMuTrain1);
						matMahalShuffled(2,1,intTrial) = vecXY1' * (matCovarTrain1 \ vecXY1);
						vecXY2 = (matTest2(:,intTrial)-vecMuTrain2);
						matMahalShuffled(2,2,intTrial) = vecXY2' * (matCovarTrain2 \ vecXY2);
					end
					matMahalShuffled(matMahalShuffled==0)=nan;
					[dummy,vecDecodedIndexMD]=min(matMahalShuffled,[],3);
					
					%decoding accuracy
					matMahalDistsFoldsShuffled = cat(3,matMahalDistsFoldsShuffled,matMahalShuffled);
				end
			end
			matMahalDistsIters = cat(4,matMahalDistsIters,matMahalDistsFolds);
			matMahalDistsItersShuffled = cat(4,matMahalDistsItersShuffled,matMahalDistsFoldsShuffled);
		end
		matMahalDists = cat(5,matMahalDists,matMahalDistsIters);
		matMahalDistsShuffled = cat(5,matMahalDistsShuffled,matMahalDistsItersShuffled);
	end
end
