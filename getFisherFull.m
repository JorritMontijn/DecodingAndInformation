function [vecI_LogReg_CV,sOut] = getFisherFull(matData,vecTrialStimType,dblLambda,dblDiffTheta,boolBiasCorrection)
	%getFisherFull Does Fisher on full matrix. Syntax:
	%    [matI_LogReg_bc_CV,sOut] = getFisherFull(matData,vecTrialStimType)
	%Format of output matrices is [intResamplings x 1 x intFolds]; standard
	%information measure (LR-based bias-corrected 10-fold CV) is first
	%output; other variations are fields of second output (non
	%bias-corrected, non-CV, activation matrix, direct, as well as shuffled
	%versions)
	%
	%	By Jorrit Montijn, 01-05-18 (dd-mm-yy; Universite de Geneve)
	
	
	
	%% general data (not class-specific)
	intFoldK = 10;
	intRepetitions = numel(vecTrialStimType)/2;
	intNeurons = size(matData,2);
	intTrialsPerFold = floor(min(intRepetitions)/intFoldK); %take lowest number of repetitions for the two classes, and round down when dividing in equal K-folds
	if intNeurons > intRepetitions
		fprintf('WARNING: Number of repetitions (%d) is less than number of neurons (%d)\n',...
			intRepetitions,intNeurons);end
	
	if ~exist('dblLambda','var'),dblLambda=0;end
	if ~exist('dblDiffTheta','var'),dblDiffTheta=2;end
	if ~exist('boolBiasCorrection','var'),boolBiasCorrection=true;end 
	boolLinear=false;
	
	%% pre-allocate
	vecA_LogReg = nan(intFoldK,1);
	vecI_LogReg_CV = nan(intFoldK,1);
	vecI_LogReg = nan(intFoldK,1);
	vecI_Direct_bc = nan(intFoldK,1);
	
	matA_LogReg_shuff = nan(intFoldK,1);
	vecI_LogReg_CV_shuff = nan(intFoldK,1);
	vecI_LogReg_shuff = nan(intFoldK,1);
	vecI_Direct_bc_shuff = nan(intFoldK,1);
	
	%% select data
	matData1 = matData(vecTrialStimType==1,:)';
	matData2 = matData(vecTrialStimType==2,:)';
	
	%% build folds
	[cellFolds1,cellFolds2,cellFoldsShuffled1,cellFoldsShuffled2] = getFolds(intFoldK,matData1,matData2);
	
	
	%% run
	%parfor intFold=1:intFoldK
	for intFold=1:intFoldK
		
		%% get correction factors
		intGroupSize = intNeurons;
		intTrials12 = (intTrialsPerFold)*(intFoldK-1); %check if this one
		dblSubFac =(2*intGroupSize)/(intTrials12*(dblDiffTheta.^2));
		dblProdFacRaw = ((2*intTrials12-intGroupSize-3)/(2*intTrials12-2));
		if ~boolBiasCorrection
			dblSubFac = 0;
			dblProdFacRaw = 1;
		end
		
		%% non-shuffled
		%get training & test set
		indFolds = true(1,intFoldK);
		indFolds(intFold) = false;
		matTrain1 = cell2mat(cellFolds1(indFolds));
		matTest1 = cell2mat(cellFolds1(~indFolds));
		matTrain2 = cell2mat(cellFolds2(indFolds));
		matTest2 = cell2mat(cellFolds2(~indFolds));
		
		%get logistic regression output
		[vecWeightsLogReg, dblLLH] = doBinLogReg([matTrain1 matTrain2], [zeros(1,size(matTrain1,2)) ones(1,size(matTrain2,2))], dblLambda);
		
		%non-CV
		vecClass1NonCV = vecWeightsLogReg'*[matTrain1;ones(1,size(matTrain1,2))];
		vecClass2NonCV = vecWeightsLogReg'*[matTrain2;ones(1,size(matTrain2,2))];
		dblDprimeLogReg = getdprime2(vecClass1NonCV,vecClass2NonCV);
		%CV
		vecClass1 = vecWeightsLogReg'*[matTest1;ones(1,size(matTest1,2))];
		vecClass2 = vecWeightsLogReg'*[matTest2;ones(1,size(matTest2,2))];
		dblDprimeLogRegCV = getdprime2(vecClass1,vecClass2);
		vecClassIdx = [ones(1,size(matTest1,2)) 2*ones(1,size(matTest1,2))];
		
		%save
		vecI_LogReg_CV(intFold) = (dblDprimeLogRegCV.^2)/(dblDiffTheta^2);%*dblProdFacRaw-dblSubFac;
		vecI_LogReg(intFold) = (dblDprimeLogReg.^2)/(dblDiffTheta^2);%*dblProdFacRaw-dblSubFac;
		
		%get direct output
		[dblPredA,matPredA,dblI,dblImat,dblI_diag] = getSeparation([matTrain1 matTrain2],[zeros(1,size(matTrain1,2)) ones(1,size(matTrain2,2))],boolLinear,dblDiffTheta);
		vecI_Direct_bc(intFold) = dblI*dblProdFacRaw-dblSubFac;
		
		%% shuffled
		%get training & test set
		matTrain1 = cell2mat(cellFoldsShuffled1(indFolds));
		matTest1 = cell2mat(cellFoldsShuffled1(~indFolds));
		matTrain2 = cell2mat(cellFoldsShuffled2(indFolds));
		matTest2 = cell2mat(cellFoldsShuffled2(~indFolds));
		
		%get logistic regression output
		[vecWeightsLogReg, dblLLH] = doBinLogReg([matTrain1 matTrain2], [zeros(1,size(matTrain1,2)) ones(1,size(matTrain2,2))], dblLambda);
		
		%non-CV
		vecClass1NonCV = vecWeightsLogReg'*[matTrain1;ones(1,size(matTrain1,2))];
		vecClass2NonCV = vecWeightsLogReg'*[matTrain2;ones(1,size(matTrain2,2))];
		dblDprimeLogReg = getdprime2(vecClass1NonCV,vecClass2NonCV);
		%CV
		vecClass1 = vecWeightsLogReg'*[matTest1;ones(1,size(matTest1,2))];
		vecClass2 = vecWeightsLogReg'*[matTest2;ones(1,size(matTest2,2))];
		dblDprimeLogRegCV = getdprime2(vecClass1,vecClass2);
		vecClassIdx = [ones(1,size(matTest1,2)) 2*ones(1,size(matTest1,2))];
		
		%save
		vecI_LogReg_CV_shuff(intFold) = (dblDprimeLogRegCV.^2)/(dblDiffTheta^2);%*dblProdFacRaw-dblSubFac;
		vecI_LogReg_shuff(intFold) = (dblDprimeLogReg.^2)/(dblDiffTheta^2);%*dblProdFacRaw-dblSubFac;
		
		%get direct output
		[dblPredA,matPredA,dblI,dblImat,dblI_diag] = getSeparation([matTrain1 matTrain2],[zeros(1,size(matTrain1,2)) ones(1,size(matTrain2,2))],boolLinear,dblDiffTheta);
		vecI_Direct_bc_shuff(intFold) = dblI*dblProdFacRaw-dblSubFac;
	end
	
	
	
	
	%% save
	%put in output
	if nargout > 1
		sOut = struct;
		sOut.matA_LogReg = matA_LogReg;
		sOut.matI_LogReg_bc_CV = matI_LogReg_CV;
		sOut.matI_LogReg_bc = matI_LogReg;
		sOut.matI_Direct_bc = matI_Direct_bc;
		
		sOut.matA_LogReg_shuff = matA_LogReg_shuff;
		sOut.matI_LogReg_bc_CV_shuff = matI_LogReg_CV_shuff;
		sOut.matI_LogReg_bc_shuff = matI_LogReg_shuff;
		sOut.matI_Direct_bc_shuff = matI_Direct_bc_shuff;
	end
	%end
	
