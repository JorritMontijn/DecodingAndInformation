function sOut = doFisherAnalTrials(matData,vecTrialStimType,sParams)
	%doFisherAnal Does Fisher Anal. Syntax:
	%   sOut = doFisherAnal(matData,vecTrialStimType,sParams)
	%Input fields:
	%	intFoldK		(10)
	%	vecUseStimTypes ([1 2])
	%	vecGroupSizes	(trial dependent)
	%	dblLambda		(0)
	%	boolDirectI		(false)
	%	boolVerbose		(true)
	%	dblDiffTheta	(1)
	%Output fields:
	%	vecI
	%	vecI_shuff
	%	matI_LogReg_bc_CV
	%	matI_LogReg_bc
	%	matI_Direct_bc
	%	matI_LogReg_bc_CV_shuff
	%	matI_LogReg_bc_shuff
	%	matI_Direct_bc_shuff
	%	vecGroupSizes
	%	vecUseStimTypes
	%
	%	By Jorrit Montijn, 25-04-17 (dd-mm-yy; Universite de Geneve)
	
	
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
	if isfield(sParams,'dblLambda'),dblLambda=sParams.dblLambda;else dblLambda = 0;end
	if isfield(sParams,'intIters'),intIters=sParams.intIters;else intIters = 20;end
	if isfield(sParams,'boolDirectI'),boolDirectI=sParams.boolDirectI;else boolDirectI = false;end
	if isfield(sParams,'boolVerbose'),boolVerbose=sParams.boolVerbose;else boolVerbose = true;end
	if isfield(sParams,'dblDiffTheta'),dblDiffTheta=sParams.dblDiffTheta;else dblDiffTheta = 1;end
	if isfield(sParams,'boolDecodeML'),boolDecodeML=sParams.boolDecodeML;else boolDecodeML = false;end
	if isfield(sParams,'boolDecodeLR'),boolDecodeLR=sParams.boolDecodeLR;else boolDecodeLR = false;end
	if isfield(sParams,'boolDecodeMD'),boolDecodeMD=sParams.boolDecodeMD;else boolDecodeMD = false;end
	
	
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
	
	%% pre-allocate
	matA_LogReg = nan(intFoldK,intTrialsPerFold,2);
	matI_LogReg_bc_CV = nan(intFoldK,intTrialsPerFold,2);
	
	matA_LogReg_shuff = nan(intFoldK,intTrialsPerFold,2);
	matI_LogReg_bc_CV_shuff = nan(intFoldK,intTrialsPerFold,2);
	
	%% run
	for intFold=1:intFoldK
		%% get correction factors
		intTrials12 = (intTrialsPerFold)*(intFoldK-1); %check if this one
		dblSubFac =(2*intNeurons)/(intTrials12*(dblDiffTheta.^2));
		dblProdFacRaw = ((2*intTrials12-intNeurons-3)/(2*intTrials12-2));
		
		%% non-shuffled
		%get training & test set
		indFolds = true(1,intFoldK);
		indFolds(intFold) = false;
		matTrain1 = cell2mat(cellFolds1(indFolds));
		%matTrain1 = matTrain1(vecNeurons,:);
		matTest1 = cell2mat(cellFolds1(~indFolds));
		%matTest1 = matTest1(vecNeurons,:);
		matTrain2 = cell2mat(cellFolds2(indFolds));
		%matTrain2 = matTrain2(vecNeurons,:);
		matTest2 = cell2mat(cellFolds2(~indFolds));
		%matTest2 = matTest2(vecNeurons,:);
		
		%get logistic regression output
		[vecWeightsLogReg, dblLLH] = doBinLogReg([matTrain1 matTrain2], [zeros(1,size(matTrain1,2)) ones(1,size(matTrain2,2))], dblLambda);
		
		%CV
		vecClass1 = vecWeightsLogReg'*[matTest1;ones(1,size(matTest1,2))];
		vecClass2 = vecWeightsLogReg'*[matTest2;ones(1,size(matTest2,2))];
		vecDprime1 = getdprimevec(vecClass1,vecClass2);
		vecA1 = sum(matTest1,1);
		vecDprime2 = getdprimevec(vecClass2,vecClass1);
		vecA2 = sum(matTest2,1);
		
		%save
		matI_LogReg_bc_CV(intFold,:,1) = (vecDprime1.^2)*dblProdFacRaw-dblSubFac;
		matA_LogReg(intFold,:,1) = vecA1;
		matI_LogReg_bc_CV(intFold,:,2) = (vecDprime2.^2)*dblProdFacRaw-dblSubFac;
		matA_LogReg(intFold,:,2) = vecA2;
		
		%% shuffled
		%get training & test set
		matTrain1 = cell2mat(cellFoldsShuffled1(indFolds));
		%matTrain1 = matTrain1(vecNeurons,:);
		matTest1 = cell2mat(cellFoldsShuffled1(~indFolds));
		%matTest1 = matTest1(vecNeurons,:);
		matTrain2 = cell2mat(cellFoldsShuffled2(indFolds));
		%matTrain2 = matTrain2(vecNeurons,:);
		matTest2 = cell2mat(cellFoldsShuffled2(~indFolds));
		%matTest2 = matTest2(vecNeurons,:);
		
		%get logistic regression output
		[vecWeightsLogReg, dblLLH] = doBinLogReg([matTrain1 matTrain2], [zeros(1,size(matTrain1,2)) ones(1,size(matTrain2,2))], dblLambda);
		
		%CV
		vecClass1 = vecWeightsLogReg'*[matTest1;ones(1,size(matTest1,2))];
		vecClass2 = vecWeightsLogReg'*[matTest2;ones(1,size(matTest2,2))];
		vecDprime1_s = getdprimevec(vecClass1,vecClass2);
		vecA1_s = sum(matTest1,1);
		vecDprime2_s = getdprimevec(vecClass2,vecClass1);
		vecA2_s = sum(matTest2,1);
		
		%save
		matI_LogReg_bc_CV_shuff(intFold,:,1) = (vecDprime1_s.^2)*dblProdFacRaw-dblSubFac;
		matA_LogReg_shuff(intFold,:,1) = vecA1_s;
		matI_LogReg_bc_CV_shuff(intFold,:,2) = (vecDprime2_s.^2)*dblProdFacRaw-dblSubFac;
		matA_LogReg_shuff(intFold,:,2) = vecA2_s;
		
		
	end
	
	%% save
	%put in output
	sOut = struct;
	sOut.matA_LogReg = matA_LogReg;
	sOut.matI_LogReg_bc_CV = matI_LogReg_bc_CV;
	sOut.matA_LogReg_shuff = matA_LogReg_shuff;
	sOut.matI_LogReg_bc_CV_shuff = matI_LogReg_bc_CV_shuff;
	sOut.vecUseStimTypes = vecUseStimTypes;
	
	%end
	
