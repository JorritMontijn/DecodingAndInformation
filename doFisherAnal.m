function [matFisherFull,sAgg] = doFisherAnal(matData,vecTrialStimType,sParamsAnal)
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
	
	%% start
	vecGroupSizes = sParamsAnal.vecGroupSizes;
	intIters = sParamsAnal.intIters;
	dblLambda = sParamsAnal.dblLambda;
	dblDiffTheta = sParamsAnal.dblDiffTheta;
	vecUseStimTypes = sParamsAnal.vecUseStimTypes;
	if isfield(sParamsAnal,'boolBiasCorrection'),boolBiasCorrection = sParamsAnal.boolBiasCorrection;else boolBiasCorrection = true;end
	if isfield(sParamsAnal,'boolDirectI'),boolDirectI = sParamsAnal.boolDirectI;else boolDirectI = true;end
	if isfield(sParamsAnal,'boolLogRegI'),boolLogRegI=sParamsAnal.boolLogRegI;else boolLogRegI = true;end
	
	%set parameters
	matFisherFull = [];
	for intGroupIdx=1:numel(vecGroupSizes)
		dblGroupSize = vecGroupSizes(intGroupIdx);
		sParamsAnalSplit = struct;
		sParamsAnalSplit.intSizeX = dblGroupSize;
		sParamsAnalSplit.intSizeY = 0;
		sParamsAnalSplit.vecUseStimTypes=vecUseStimTypes;
		sParamsAnalSplit.intResamplings = intIters;
		
		%get splits
		[cellMatX,cellNeuronsX,cellMatY,cellNeuronsY] = doDimDataSplits(matData,vecTrialStimType,sParamsAnalSplit);
		fprintf('  Created %dx%d data splits; group size %d (%d/%d) [%s]\n',size(cellMatX),dblGroupSize,intGroupIdx,numel(vecGroupSizes),getTime);
		
		%set params
		sParams = struct;
		sParams.dblLambda = dblLambda;
		sParams.boolDirectI = boolDirectI;
		sParams.boolLogRegI = boolLogRegI;
		sParams.boolVerbose = false;
		sParams.dblDiffTheta = dblDiffTheta;
		sParams.boolBiasCorrection = boolBiasCorrection;
		
		%% get Fisher info with full pop
		[matFisherThis,sOut] = doFisherFull(cellMatX,[],sParams);
		matFisherFull = cat(2,matFisherFull,matFisherThis);
		if ~exist('sAgg','var')
			sAgg = sOut;
			sFields = fieldnames(sOut);
		else %add output from multiple group sizes together iteratively
			for intField=1:numel(sFields)
				sAgg.(sFields{intField}) = cat(2,sAgg.(sFields{intField}),sOut.(sFields{intField}));
			end
		end
	end
	%end
	
