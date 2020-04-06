function sOut = doFisherAnalTrialsSubSample(matData,vecTrialStimType,sParams,intNumberOfSpikes)
	%doFisherAnalTrialsSubSample Calculated Fisher information using the
	%number of neurons that yields a number of spikes per trial closest to
	%the one requested as input (intNumberOfSpikes) 
	%   sOut = doFisherAnalTrialsSubSample(matData,vecTrialStimType,sParams,intNumberOfSpikes)
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
	if isfield(sParams,'boolVerbose'),boolVerbose=sParams.boolVerbose;else boolVerbose = true;end
	if isfield(sParams,'dblDiffTheta'),dblDiffTheta=sParams.dblDiffTheta;else dblDiffTheta = 1;end
	
	
	%% general data (not class-specific)
	intGroups = numel(vecGroupSizes);
	intNeurons = size(matData,1);
	
	%% prep data (12-class-specific)
	indKeepTrials = ismember(vecTrialStimType,vecUseStimTypes);
	vecClasses12 = label2idx(vecTrialStimType(indKeepTrials));
	matData12 =  matData(:,indKeepTrials);
	matData1 = matData12(:,vecClasses12==1);
	matData2 = matData12(:,vecClasses12==2);
	intTrials12 = size(matData1,2);
	
	%% pre-allocate
	matN = nan(intTrials12,2); %number of neurons
	matA = nan(intTrials12,2);
	matA_LogReg = nan(intTrials12,2);
	matI_LogReg_bc_CV = nan(intTrials12,2);

	%% run
	for intTrial=1:intTrials12
		%% get data
		if boolVerbose && mod(intTrial,100) == 0,fprintf('Now at trial %d/%d [%s]\n',intTrial,intTrials12,getTime);end
		for intStim=1:2
			%% get activity for this trial
			if intStim==1
				vecAct = matData1(:,intTrial);
			else
				vecAct = matData2(:,intTrial);
			end
			
			%% get selection vector of neurons closest to required number of spikes
			if sum(vecAct) < intNumberOfSpikes,continue;end
			intEstimatedReqN = round(intNumberOfSpikes/(sum(vecAct)/intNeurons));
			vecNeurons = randperm(intNeurons,intEstimatedReqN);
			intNumSpikesT = sum(vecAct(vecNeurons));
			if intNumSpikesT == intNumberOfSpikes %do nothing
			elseif intNumSpikesT < intNumberOfSpikes
				while intNumSpikesT < intNumberOfSpikes
					vecUnused = find(~ismember(1:intNeurons,vecNeurons));
					vecNeurons(end+1) = vecUnused(randi(numel(vecUnused))); %#ok<AGROW>
					intNumSpikesT = sum(vecAct(vecNeurons));
				end
			elseif intNumSpikesT > intNumberOfSpikes
				while intNumSpikesT > intNumberOfSpikes
					vecNeurons(randi(numel(vecNeurons))) = [];
					intNumSpikesT = sum(vecAct(vecNeurons));
				end
			end
			intSelectedNeurons = numel(vecNeurons);
			vecSelectedAct = vecAct(vecNeurons);
			
			
			%% fit logistic regression
			%get data
			indAllOtherTrials = true(1,intTrials12);
			indAllOtherTrials(intTrial) = false;
			matTrain1 = matData1(vecNeurons,indAllOtherTrials);
			matTrain2 = matData2(vecNeurons,indAllOtherTrials);
			
			%get logistic regression output
			[vecWeightsLogReg, dblLLH] = doBinLogReg([matTrain1 matTrain2], [zeros(1,size(matTrain1,2)) ones(1,size(matTrain2,2))], dblLambda);
			dblLogAct = vecWeightsLogReg'*[vecSelectedAct;ones(1,size(vecSelectedAct,2))];
			
			%output
			matN(intTrial,intStim) = intSelectedNeurons;
			matA(intTrial,intStim) = intNumSpikesT;
			matA_LogReg(intTrial,intStim) = dblLogAct;
		end
	end
			
	%% calculate Fisher Information
	vecDprime1 = abs(getdprimevec(matA_LogReg(:,1),matA_LogReg(:,2)));
	vecDprime2 = abs(getdprimevec(matA_LogReg(:,2),matA_LogReg(:,1)));
	
	vecSubFac1 = (2*matN(:,1))/(intTrials12*(dblDiffTheta.^2));
	vecSubFac2 = (2*matN(:,2))/(intTrials12*(dblDiffTheta.^2));
	
	vecProdFacRaw1 = ((2*intTrials12-matN(:,1)-3)/(2*intTrials12-2));
	vecProdFacRaw2 = ((2*intTrials12-matN(:,2)-3)/(2*intTrials12-2));
	
	vecI_bc1 = (vecDprime1.^2).*vecProdFacRaw1-vecSubFac1;
	vecI_bc2 = (vecDprime2.^2).*vecProdFacRaw2-vecSubFac2;
	
	%% assign
	matI_LogReg_bc_CV(:,1) = vecI_bc1;
	matI_LogReg_bc_CV(:,2) = vecI_bc2;
	
	%% save
	%put in output
	sOut = struct;
	sOut.matN = matN;
	sOut.matA = matA;
	sOut.matA_LogReg = matA_LogReg;
	sOut.matI_LogReg_bc_CV = matI_LogReg_bc_CV;
	sOut.vecUseStimTypes = vecUseStimTypes;
	sOut.vecClasses12 = vecClasses12;
	
	
	%end
	
