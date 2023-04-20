
%% generate data
%stim params
intReps = 20;
vecDirs = 0:15:359;
dblStimDur = 1;
dblITI = 0.5;
dblDur = dblStimDur+dblITI;
vecTrialAngles = repmat(vecDirs,[1 intReps]);
dblEdgeDur = 10;
vecOnsets = dblEdgeDur:dblDur:(dblEdgeDur + dblDur*(numel(vecTrialAngles)-0.1));
vecOffsets = vecOnsets + dblStimDur;
matTrialT = cat(2,vecOnsets',vecOffsets');

%neuron params
intNeuronNum = 50;
vecBaseRate = 1+exprnd(1,[1 intNeuronNum]);
vecPrefRate = vecBaseRate+(vecBaseRate.*exprnd(1,[1 intNeuronNum]));
vecKappa = 5 + 5*rand([1 intNeuronNum]);
boolDoublePeaked = false;
vecPrefOri = 2*pi*rand([1 intNeuronNum]);
cellSpikeTimes = cell(1,intNeuronNum);
for intNeuron=1:intNeuronNum
	dblBaseRate = vecBaseRate(intNeuron);
	dblPrefRate = vecPrefRate(intNeuron);
	dblKappa = vecKappa(intNeuron);
	dblPrefOri = vecPrefOri(intNeuron);
	cellSpikeTimes{intNeuron} = getGeneratedSpikingData(...
		deg2rad(vecTrialAngles),matTrialT,dblBaseRate,dblPrefRate,dblKappa,boolDoublePeaked,dblPrefOri);
	
	
end

% build resp mat
matSpikeCounts = getSpikeCounts(cellSpikeTimes,vecOnsets,vecOffsets);

%% test 1
for intRand=0:1;
	figure;drawnow;
	if intRand == 0
		vecTrialTypes = vecTrialAngles;
	else
		vecTrialTypes = vecTrialTypes(randperm(numel(vecTrialAngles)));
	end
	vecNeuronsX = 1:(intNeuronNum-1);%:round((intNeuronNum/2));
	vecNeuronsY = find(~ismember(1:intNeuronNum,vecNeuronsX));
	[vecTrialTypeIdx,vecUniqueTrialTypes,vecCounts,cellSelect,vecRepetition] = val2idx(vecTrialTypes);
	varTypeCV = 2;
	dblLambda = 1;
	
	%old ML
	tic
	[dblPerformanceCV_ML,vecDecodedIndexCV_ML,matPosteriorProbabilityCV_ML,dblMeanErrorDegs_ML,matConfusion_ML] ...
		= doCrossValidatedDecodingML(matSpikeCounts,vecTrialTypes,varTypeCV);
	toc
	subplot(2,3,1)
	imagesc(matConfusion_ML)
	title('ML, repetition-wise CV')
	
	%old LR
	tic
	[dblPerformanceCV_LR,vecDecodedIndexCV_LR,matPosteriorProbability_LR,dblMeanErrorDegs_LR,matConfusion_LR] ...
		= doCrossValidatedDecodingLR(matSpikeCounts,vecTrialTypes,varTypeCV,[],dblLambda);
	toc
	subplot(2,3,2)
	imagesc(matConfusion_LR)
	title('LR, repetition-wise CV')
	
	
	%new mvn
	tic
	[dblPerformanceCV_Mvn,vecDecodedIndexCV_Mvn,matPosteriorProbability_Mvn,dblMeanErrorDegs_Mvn,matConfusion_Mvn] ...
		= doCrossValidatedDecoding(matSpikeCounts,vecTrialTypes,varTypeCV,vecCounts,dblLambda);
	toc
	subplot(2,3,3)
	imagesc(matConfusion_Mvn)
	title('Mvn, repetition-wise CV')
	
	%new mvn no CV
	tic
	[dblPerformanceCV4,vecDecodedIndexCV4,matPosteriorProbability4,dblMeanErrorDegs4,matConfusion4] ...
		= doCrossValidatedDecoding(matSpikeCounts,vecTrialTypes,0,vecCounts,dblLambda);
	toc
	subplot(2,3,4)
	imagesc(matConfusion4)
	title('Mvn, no CV')
	
	%new mvn single-trial CV
	tic
	[dblPerformanceCV5,vecDecodedIndexCV5,matPosteriorProbability5,dblMeanErrorDegs5,matConfusion5] ...
		= doCrossValidatedDecoding(matSpikeCounts,vecTrialTypes,1,vecCounts,dblLambda);
	toc
	subplot(2,3,5)
	imagesc(matConfusion5)
	title('Mvn, trial-wise CV')
	
	%new mvn k-fold
	tic
	[dblPerformanceCV6,vecDecodedIndexCV6,matPosteriorProbability6,dblMeanErrorDegs6,matConfusion6] ...
		= doCrossValidatedDecoding(matSpikeCounts,vecTrialTypes,vecRepetition,vecCounts,dblLambda);
	toc
	subplot(2,3,6)
	imagesc(matConfusion6)
	title('Mvn, K-fold CV')
	drawnow;
end