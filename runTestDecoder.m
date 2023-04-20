
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
vecNeuronsX = 1:(intNeuronNum-1);%:round((intNeuronNum/2));
vecNeuronsY = find(~ismember(1:intNeuronNum,vecNeuronsX));
vecTrialTypes = vecTrialTypes(randperm(numel(vecTrialAngles)));
[vecTrialTypeIdx,vecUniqueTrialTypes,vecCounts,cellSelect,vecRepetition] = val2idx(vecTrialTypes);
varTypeCV = 2;
dblLambda = 1;
tic
[dblPerformanceCV,vecDecodedIndexCV,matPosteriorProbability,dblMeanErrorDegs,matConfusion,matWeights,matAggActivation,matAggWeights,vecRepetition] ...
	= doCrossValidatedDecodingLR(matSpikeCounts,vecTrialTypes,varTypeCV,[],dblLambda);
toc
figure
subplot(2,3,1)
imagesc(matConfusion)
% test 2
tic
[dblPerformanceCV2,vecDecodedIndexCV2,matPosteriorProbability2,dblMeanErrorDegs2,matConfusion2] ...
	= doCrossValidatedDecoding(matSpikeCounts,vecTrialTypes,varTypeCV,vecCounts,dblLambda);
toc
subplot(2,3,2)
imagesc(matConfusion2)

% test 3
tic
[dblPerformanceCV3,vecDecodedIndexCV3,matPosteriorProbabilityCV3,dblMeanErrorDegs3,matConfusion3] ...
	= doCrossValidatedDecodingML(matSpikeCounts,vecTrialTypes,varTypeCV);
toc
subplot(2,3,3)
imagesc(matConfusion3)

% test 4
tic
[dblPerformanceCV4,vecDecodedIndexCV2,matPosteriorProbability2,dblMeanErrorDegs2,matConfusion4] ...
	= doCrossValidatedDecoding(matSpikeCounts,vecTrialTypes,0,vecCounts,dblLambda);
toc
subplot(2,3,4)
imagesc(matConfusion4)

% test 5
tic
[dblPerformanceCV5,vecDecodedIndexCV2,matPosteriorProbability2,dblMeanErrorDegs2,matConfusion5] ...
	= doCrossValidatedDecoding(matSpikeCounts,vecTrialTypes,1,vecCounts,dblLambda);
toc
subplot(2,3,5)
imagesc(matConfusion5)

% test 6
tic
[dblPerformanceCV6,vecDecodedIndexCV2,matPosteriorProbability2,dblMeanErrorDegs2,matConfusion6] ...
	= doCrossValidatedDecoding(matSpikeCounts,vecTrialTypes,vecRepetition,vecCounts,dblLambda);
toc
subplot(2,3,6)
imagesc(matConfusion6)
