function [dblDecodedAngle] = doGlobalOriEstimator2(vecThisAct, matThisData, vecThisOriRads)
% Chengcheng's weight estimator
% matData must have dims trials x neurons.
% vecClasses must have dims trials x 1.

N = size(matThisData,1);
angIdx = label2idx(vecThisOriRads);
Nangs = size(angIdx,1);

Sigma = matThisData'*matThisData/N;
fMat = splitapply(@mean,matThisData,angIdx);
fMatfMatT = fMat'*fMat/Nangs;
Fourierf1 = mean(bsxfun(@times,exp(-1i * vecThisOriRads),matThisData),1);

vecWeights = (Sigma+fMatfMatT)\Fourierf1;
vecWeights = conj(vecWeights);

dblDecodedAngle = angle(vecThisAct * vecWeights);
end
