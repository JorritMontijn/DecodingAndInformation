function sOut = doFisherAnal(matData,vecTrialStimType,sParams)
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
	matI_LogReg_bc_CV = nan(intIters,intGroups,intFoldK);
	matI_LogReg_bc = nan(intIters,intGroups,intFoldK);
	if boolDirectI,matI_Direct_bc = nan(intIters,intGroups,intFoldK);end
	if boolDecodeLR,matA_LR = nan(intIters,intGroups,intFoldK);end
	if boolDecodeML,matA_ML = nan(intIters,intGroups,intFoldK);end
	if boolDecodeMD,matA_MD = nan(intIters,intGroups,intFoldK);end
	
	matI_LogReg_bc_CV_shuff = nan(intIters,intGroups,intFoldK);
	matI_LogReg_bc_shuff = nan(intIters,intGroups,intFoldK);
	if boolDirectI,matI_Direct_bc_shuff = nan(intIters,intGroups,intFoldK);end
	if boolDecodeLR,matA_LR_shuff = nan(intIters,intGroups,intFoldK);end
	if boolDecodeML,matA_ML_shuff = nan(intIters,intGroups,intFoldK);end
	if boolDecodeMD,matA_MD_shuff = nan(intIters,intGroups,intFoldK);end
	
	%% run
	for intGroupSizeIdx=1:intGroups
		intGroupSize = vecGroupSizes(intGroupSizeIdx);
		if intGroupSize > 100
			intIters = min([20 intIters]);
		end
		if boolVerbose,fprintf('Doing Fisher Anal; now at group size %d (%d/%d) [%s]\n',intGroupSize,intGroupSizeIdx,intGroups,getTime);end
		for intIter=1:intIters
			vecNeurons = randperm(intNeurons,intGroupSize);
			vecNeurons =1:intNeurons;
			if boolVerbose,fprintf('   Iteration %d/%d [%s]\n',intIter,intIters,getTime);end
		
			for intFold=1:intFoldK
				%% get correction factors
				intTrials12 = (intTrialsPerFold)*(intFoldK-1); %check if this one
				dblSubFac =(2*intGroupSize)/(intTrials12*(dblDiffTheta.^2));
				dblProdFacRaw = ((2*intTrials12-intGroupSize-3)/(2*intTrials12-2));
				
				%% non-shuffled
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
				matI_LogReg_bc_CV(intIter,intGroupSizeIdx,intFold) = (dblDprimeLogRegCV.^2)*dblProdFacRaw-dblSubFac;
				matI_LogReg_bc(intIter,intGroupSizeIdx,intFold) = (dblDprimeLogReg.^2)*dblProdFacRaw-dblSubFac;
				
				%get direct output
				if boolDirectI
					[dblPredA,matPredA,dblD2,dblD2mat,dblD2_diag] = getSeparation([matTrain1 matTrain2],[zeros(1,size(matTrain1,2)) ones(1,size(matTrain2,2))],0);
					matI_Direct_bc(intIter,intGroupSizeIdx,intFold) = dblD2*dblProdFacRaw-dblSubFac;
				end
				
				%get accuracy for logistic regression
				if boolDecodeLR
					%get performance
					matActivation = [0*vecClass1 0*vecClass2; vecClass1 vecClass2];
					matPosteriorProbability = exp(bsxfun(@minus,matActivation,logsumexp(matActivation,1))); %softmax
					[dummy, vecDecodedIndexLR] = max(matPosteriorProbability,[],1);
					%decoding accuracy
					matA_LR(intIter,intGroupSizeIdx,intFold) = sum(vecDecodedIndexLR(:)==vecClassIdx(:))/numel(vecDecodedIndexLR);
				end
				
				%get accuracy for maximum-likelihood
				if boolDecodeML
					%get likelihood on train set
					matParametersML = nan(intGroupSize,2,2);
					
					%put data in likelihood parameter matrix for type1
					matParametersML(:,1,1) = xmean(matTrain1,2);
					matParametersML(:,1,2) = xstd(matTrain1,2);
					
					%put data in likelihood parameter matrix for type2
					matParametersML(:,2,1) = xmean(matTrain2,2);
					matParametersML(:,2,2) = xstd(matTrain2,2);
					intTestT1 = size(matTest1,2);
					
					% test on test set
					matLikelihood = nan([intGroupSize size(matTest1,2)+size(matTest2,2) 2]);
					for intTrial = 1:intTestT1
						matLikelihood(:,intTrial,1) = normpdf(matTest1(:,intTrial), matParametersML(:,1,1),matParametersML(:,1,2));
						matLikelihood(:,intTrial,2) = normpdf(matTest1(:,intTrial), matParametersML(:,2,1),matParametersML(:,2,2));
					end
					for intTrial = 1:size(matTest2,2)
						matLikelihood(:,intTrial+intTestT1,1) = normpdf(matTest2(:,intTrial), matParametersML(:,1,1),matParametersML(:,1,2));
						matLikelihood(:,intTrial+intTestT1,2) = normpdf(matTest2(:,intTrial), matParametersML(:,2,1),matParametersML(:,2,2));
					end
					matLikelihood(matLikelihood==0)=nan;
					[dummy,vecDecodedIndexML]=min(nansum(-log(matLikelihood),1),[],3);
					
					%decoding accuracy
					matA_ML(intIter,intGroupSizeIdx,intFold) = sum(vecDecodedIndexML(:)==vecClassIdx(:))/numel(vecDecodedIndexML);
				end
				
				%get accuracy for multidimensional decoder
				if boolDecodeMD
					%no CV
					intTestT1 = size(matTest1,2);
					vecMuTrain1 = xmean(matTrain1,2);
					matCovarTrain1 = cov(matTrain1');
					vecMuTrain2 = xmean(matTrain2,2);
					matCovarTrain2 = cov(matTrain2');
					
					%mahal
					matMahal=nan([1 size(matTest1,2)+size(matTest2,2) 2]);
					for intTrial = 1:intTestT1 %can be used as parfor
						vecXY1 = (matTest1(:,intTrial)-vecMuTrain1);
						matMahal(1,intTrial,1) = vecXY1' * (matCovarTrain1 \ vecXY1);
						vecXY2 = (matTest1(:,intTrial)-vecMuTrain2);
						matMahal(1,intTrial,2) = vecXY2' * (matCovarTrain2 \ vecXY2);
					end
					for intTrial = (intTestT1+1):size(matMahal,2)
						vecXY1 = (matTest2(:,intTrial-intTestT1)-vecMuTrain1);
						matMahal(1,intTrial,1) = vecXY1' * (matCovarTrain1 \ vecXY1);
						vecXY2 = (matTest2(:,intTrial-intTestT1)-vecMuTrain2);
						matMahal(1,intTrial,2) = vecXY2' * (matCovarTrain2 \ vecXY2);
					end
					matMahal(matMahal==0)=nan;
					[dummy,vecDecodedIndexMD]=min(matMahal,[],3);
					
					%decoding accuracy
					matA_MD(intIter,intGroupSizeIdx,intFold) = sum(vecDecodedIndexMD(:)==vecClassIdx(:))/numel(vecDecodedIndexMD);
				end
				
				
				%% shuffled
				%get training & test set
				matTrain1 = cell2mat(cellFoldsShuffled1(indFolds));
				matTrain1 = matTrain1(vecNeurons,:);
				matTest1 = cell2mat(cellFoldsShuffled1(~indFolds));
				matTest1 = matTest1(vecNeurons,:);
				matTrain2 = cell2mat(cellFoldsShuffled2(indFolds));
				matTrain2 = matTrain2(vecNeurons,:);
				matTest2 = cell2mat(cellFoldsShuffled2(~indFolds));
				matTest2 = matTest2(vecNeurons,:);
				
				%get logistic regression output
				[vecWeightsLogReg, dblLLH] = doBinLogReg([matTrain1 matTrain2], [zeros(1,size(matTrain1,2)) ones(1,size(matTrain2,2))], dblLambda);
				
				%non-CV
				vecClass1NonCV = vecWeightsLogReg'*[matTrain1;ones(1,size(matTrain1,2))];
				vecClass2NonCV = vecWeightsLogReg'*[matTrain2;ones(1,size(matTrain2,2))];
				dblDprimeLogReg_shuff = getdprime2(vecClass1NonCV,vecClass2NonCV);
				%CV
				vecClass1 = vecWeightsLogReg'*[matTest1;ones(1,size(matTest1,2))];
				vecClass2 = vecWeightsLogReg'*[matTest2;ones(1,size(matTest2,2))];
				dblDprimeLogRegCV_shuff = getdprime2(vecClass1,vecClass2);
				vecClassIdx = [ones(1,size(matTest1,2)) 2*ones(1,size(matTest1,2))];
				
				%save
				matI_LogReg_bc_CV_shuff(intIter,intGroupSizeIdx,intFold) = (dblDprimeLogRegCV_shuff.^2)*dblProdFacRaw-dblSubFac;
				matI_LogReg_bc_shuff(intIter,intGroupSizeIdx,intFold) = (dblDprimeLogReg_shuff.^2)*dblProdFacRaw-dblSubFac;
				
				%get direct output
				if boolDirectI
					[dblPredA,matPredA,dblD2_shuff,dblD2mat,dblD2_diag] = getSeparation([matTrain1 matTrain2],[zeros(1,size(matTrain1,2)) ones(1,size(matTrain2,2))],0);
					matI_Direct_bc_shuff(intIter,intGroupSizeIdx,intFold) = dblD2_shuff*dblProdFacRaw-dblSubFac;
				end
				
				%get accuracy
				if boolDecodeLR
					%get performance
					matActivation = [0*vecClass1 0*vecClass2; vecClass1 vecClass2];
					matPosteriorProbability = exp(bsxfun(@minus,matActivation,logsumexp(matActivation,1))); %softmax
					[dummy, vecDecodedIndex] = max(matPosteriorProbability,[],1);
					
					%decoding accuracy
					matA_LR_shuff(intIter,intGroupSizeIdx,intFold) = sum(vecDecodedIndex(:)==vecClassIdx(:))/numel(vecDecodedIndex);
				end
				
				%get accuracy for maximum-likelihood
				if boolDecodeML
					%get likelihood on train set
					matParametersML = nan(intGroupSize,2,2);
					
					%put data in likelihood parameter matrix for type1
					matParametersML(:,1,1) = xmean(matTrain1,2);
					matParametersML(:,1,2) = xstd(matTrain1,2);
					
					%put data in likelihood parameter matrix for type2
					matParametersML(:,2,1) = xmean(matTrain2,2);
					matParametersML(:,2,2) = xstd(matTrain2,2);
					intTestT1 = size(matTest1,2);
					
					% test on test set
					matLikelihood = nan([intGroupSize size(matTest1,2)+size(matTest2,2) 2]);
					for intTrial = 1:intTestT1
						matLikelihood(:,intTrial,1) = normpdf(matTest1(:,intTrial), matParametersML(:,1,1),matParametersML(:,1,2));
						matLikelihood(:,intTrial,2) = normpdf(matTest1(:,intTrial), matParametersML(:,2,1),matParametersML(:,2,2));
					end
					for intTrial = (intTestT1+1):size(matLikelihood,2)
						matLikelihood(:,intTrial,1) = normpdf(matTest2(:,intTrial-intTestT1), matParametersML(:,1,1),matParametersML(:,1,2));
						matLikelihood(:,intTrial,2) = normpdf(matTest2(:,intTrial-intTestT1), matParametersML(:,2,1),matParametersML(:,2,2));
					end
					matLikelihood(matLikelihood==0)=nan;
					[dummy,vecDecodedIndexML]=min(nansum(-log(matLikelihood),1),[],3);

					%decoding accuracy
					matA_ML_shuff(intIter,intGroupSizeIdx,intFold) = sum(vecDecodedIndexML(:)==vecClassIdx(:))/numel(vecDecodedIndexML);
				end
				
				%get accuracy for multidimensional decoder
				if boolDecodeMD
					%no CV
					intTestT1 = size(matTest1,2);
					vecMuTrain1 = xmean(matTrain1,2);
					matCovarTrain1 = cov(matTrain1');
					vecMuTrain2 = xmean(matTrain2,2);
					matCovarTrain2 = cov(matTrain2');
					
					%mahal
					matMahal=nan([1 size(matTest1,2)+size(matTest2,2) 2]);
					for intTrial = 1:intTestT1 %can be used as parfor
						vecXY1 = (matTest1(:,intTrial)-vecMuTrain1);
						matMahal(1,intTrial,1) = vecXY1' * (matCovarTrain1 \ vecXY1);
						vecXY2 = (matTest1(:,intTrial)-vecMuTrain2);
						matMahal(1,intTrial,2) = vecXY2' * (matCovarTrain2 \ vecXY2);
					end
					for intTrial = (intTestT1+1):size(matMahal,2)
						vecXY1 = (matTest2(:,intTrial-intTestT1)-vecMuTrain1);
						matMahal(1,intTrial,1) = vecXY1' * (matCovarTrain1 \ vecXY1);
						vecXY2 = (matTest2(:,intTrial-intTestT1)-vecMuTrain2);
						matMahal(1,intTrial,2) = vecXY2' * (matCovarTrain2 \ vecXY2);
					end
					matMahal(matMahal==0)=nan;
					[dummy,vecDecodedIndexMD]=min(matMahal,[],3);
					
					%decoding accuracy
					matA_MD_shuff(intIter,intGroupSizeIdx,intFold) = sum(vecDecodedIndexMD(:)==vecClassIdx(:))/numel(vecDecodedIndexMD);
				end
			end
		end
	end
	%clearvars matModelRespP;
	
	%% save
	%get mean+sd
	vecI = nanmean(nanmean(matI_LogReg_bc_CV,1),3);
	vecI_shuff = nanmean(nanmean(matI_LogReg_bc_CV_shuff,1),3);
	
	%put in output
	sOut = struct;
	sOut.vecI = vecI;
	sOut.vecI_shuff = vecI_shuff;
	sOut.matI_LogReg_bc_CV = matI_LogReg_bc_CV;
	sOut.matI_LogReg_bc = matI_LogReg_bc;
	if boolDirectI,sOut.matI_Direct_bc = matI_Direct_bc;end
	sOut.matI_LogReg_bc_CV_shuff = matI_LogReg_bc_CV_shuff;
	sOut.matI_LogReg_bc_shuff = matI_LogReg_bc_shuff;
	if boolDirectI,sOut.matI_Direct_bc_shuff = matI_Direct_bc_shuff;end
	sOut.vecGroupSizes = vecGroupSizes;
	sOut.vecUseStimTypes = vecUseStimTypes;
	if boolDecodeLR
		sOut.matA_LR = matA_LR;
		sOut.matA_LR_shuff = matA_LR_shuff;
	end
	if boolDecodeML
		sOut.matA_ML = matA_ML;
		sOut.matA_ML_shuff = matA_ML_shuff;
	end
	if boolDecodeMD
		sOut.matA_MD = matA_MD;
		sOut.matA_MD_shuff = matA_MD_shuff;
	end
%end

