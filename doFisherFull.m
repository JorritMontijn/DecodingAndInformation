function [matI,sOut] = doFisherFull(cellMatX,~,sParams)
	%doFisherFull Does Fisher on full matrix. Syntax:
	%    [matI_LogReg_bc_CV,sOut] = doFisherFull(cellMatX,[],sParams);
	%Format of output matrices is [intResamplings x 1 x intFolds]; standard
	%information measure (LR-based bias-corrected 10-fold CV) is first
	%output; other variations are fields of second output (non
	%bias-corrected, non-CV, activation matrix, direct, as well as shuffled
	%versions)
	%
	%	By Jorrit Montijn, 25-04-17 (dd-mm-yy; Universite de Geneve)
	
	
	%% check input
	if ~exist('sParams','var'),sParams=struct;end
	if isfield(sParams,'intFoldK'),intFoldK=sParams.intFoldK;else intFoldK = 10;end
	if isfield(sParams,'dblLambda'),dblLambda=sParams.dblLambda;else dblLambda = 0;end
	if isfield(sParams,'dblDiffTheta'),dblDiffTheta=sParams.dblDiffTheta;else dblDiffTheta = 1;end
	if isfield(sParams,'boolBiasCorrection'),boolBiasCorrection=sParams.boolBiasCorrection;else boolBiasCorrection = true;end
	if isfield(sParams,'boolDirectI'),boolDirectI=sParams.boolDirectI;else boolDirectI = true;end
	if isfield(sParams,'boolLogRegI'),boolLogRegI=sParams.boolLogRegI;else boolLogRegI = true;end
	if isfield(sParams,'boolLinear'),boolLinear=sParams.boolDirectI;else boolLinear = false;end
	
	%% general data (not class-specific)
	intResamplings = size(cellMatX,1);
	intStimTypes = size(cellMatX,2);
	intRepetitions = size(cellMatX{1,1},1);
	intNeurons = size(cellMatX{1,1},2);
	intTrialsPerFold = floor(min(intRepetitions)/intFoldK); %take lowest number of repetitions for the two classes, and round down when dividing in equal K-folds
	if intNeurons > intRepetitions
		fprintf('WARNING: Number of repetitions (%d) is less than number of neurons (%d)\n',...
			intRepetitions,intNeurons);end
	
	%% pre-allocate
	matA_LogReg = nan(intResamplings,1,intFoldK);
	matI_LogReg_CV = nan(intResamplings,1,intFoldK);
	matI_LogReg = nan(intResamplings,1,intFoldK);
	matI_Direct_bc = nan(intResamplings,1,intFoldK);
	
	matA_LogReg_shuff = nan(intResamplings,1,intFoldK);
	matI_LogReg_CV_shuff = nan(intResamplings,1,intFoldK);
	matI_LogReg_shuff = nan(intResamplings,1,intFoldK);
	matI_Direct_bc_shuff = nan(intResamplings,1,intFoldK);
	
	%% run analysis
	for intResampling=1:intResamplings
		%% select data
		matData1 = cellMatX{intResampling,1};
		matData2 = cellMatX{intResampling,2};
		if (any(range(matData1,1)==0) || any(range(matData2,1)==0))
			warning([mfilename ';Range0'],'>=1 neurons have zero range, results may be inaccurate');
		end
		
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
			matTrain1 = cell2mat(cellFolds1(indFolds))';
			matTest1 = cell2mat(cellFolds1(~indFolds))';
			matTrain2 = cell2mat(cellFolds2(indFolds))';
			matTest2 = cell2mat(cellFolds2(~indFolds))';
			
			%get logistic regression output
			if boolLogRegI
				%log reg
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
				matI_LogReg_CV(intResampling,1,intFold) = (dblDprimeLogRegCV.^2)/(dblDiffTheta^2);%*dblProdFacRaw-dblSubFac;
				matI_LogReg(intResampling,1,intFold) = (dblDprimeLogReg.^2)/(dblDiffTheta^2);%*dblProdFacRaw-dblSubFac;
			end
			
			%get direct output
			if boolDirectI
				[dblPredA,matPredA,dblI,dblImat,dblI_diag] = getSeparation([matTrain1 matTrain2],[zeros(1,size(matTrain1,2)) ones(1,size(matTrain2,2))],boolLinear,dblDiffTheta);
				matI_Direct_bc(intResampling,1,intFold) = dblI*dblProdFacRaw-dblSubFac;
			end
			
			%% shuffled
			%get training & test set
			matTrain1 = cell2mat(cellFoldsShuffled1(indFolds))';
			matTest1 = cell2mat(cellFoldsShuffled1(~indFolds))';
			matTrain2 = cell2mat(cellFoldsShuffled2(indFolds))';
			matTest2 = cell2mat(cellFoldsShuffled2(~indFolds))';
			
			%get logistic regression output
			if boolLogRegI
				%log reg
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
				matI_LogReg_CV_shuff(intResampling,1,intFold) = (dblDprimeLogRegCV.^2)/(dblDiffTheta^2);%*dblProdFacRaw-dblSubFac;
				matI_LogReg_shuff(intResampling,1,intFold) = (dblDprimeLogReg.^2)/(dblDiffTheta^2);%*dblProdFacRaw-dblSubFac;
			end
			
			%get direct output
			if boolDirectI
				[dblPredA,matPredA,dblI,dblImat,dblI_diag] = getSeparation([matTrain1 matTrain2],[zeros(1,size(matTrain1,2)) ones(1,size(matTrain2,2))],boolLinear,dblDiffTheta);
				matI_Direct_bc_shuff(intResampling,1,intFold) = dblI*dblProdFacRaw-dblSubFac;
			end
		end
	end
	
	
	
	
	%% save
	if boolLogRegI
		matI = matI_LogReg_CV;
	else
		matI = matI_Direct_bc;
	end
	
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
	
