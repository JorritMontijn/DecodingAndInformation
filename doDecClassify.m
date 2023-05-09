function vecDecodedIndexCV = doDecClassify(matPosteriorProbability,vecPriorDistribution)
	%doDecClassify Do decoding classification
	%   vecDecodedIndexCV = doDecClassify(matPosteriorProbability,vecPriorDistribution)
	
	%% loop through trials and assign next most certain trial
	intTrials = size(matPosteriorProbability,2);
	vecDecodedIndexCV = nan(intTrials,1);
	indAssignedTrials = false(intTrials,1);
	matTempProbs = matPosteriorProbability;
	for intTrial=1:intTrials
		%check if we're done
		if sum(vecPriorDistribution==0)==(numel(vecPriorDistribution)-1)
			vecDecodedIndexCV(~indAssignedTrials) = find(vecPriorDistribution>0);
			break;
		end
		
		%remove trials of type that has been chosen max number
		matTempProbs(vecPriorDistribution==0,:) = nan;
		matTempProbs(:,indAssignedTrials) = nan;
		
		%calculate probability of remaining trials and types
		[vecTempProbs,vecTempDecodedIndexCV]=max(matTempProbs,[],1);
		%get 2nd most likely stim per trial
		matDist2 = matTempProbs;
		for intT2=1:intTrials
			matDist2(vecTempDecodedIndexCV(intT2),intT2) = nan;
		end
		[vecTempProbs2,vecTempDecodedIndexCV2]=max(matDist2,[],1);
		
		%use trial with largest difference between most likely and 2nd most likely stimulus
		vecMaxDiff = abs(vecTempProbs2 - vecTempProbs);
		%assign trial
		[dummy,intAssignTrial]=max(vecMaxDiff);
		intAssignType = vecTempDecodedIndexCV(intAssignTrial);
		if vecPriorDistribution(intAssignType) == 0
			intAssignType = vecTempDecodedIndexCV2(intAssignTrial);
		end
		vecDecodedIndexCV(intAssignTrial) = intAssignType;
		indAssignedTrials(intAssignTrial) = true;
		vecPriorDistribution(intAssignType) = vecPriorDistribution(intAssignType) - 1;
		%fprintf('assigned %d to %d; %s\n',intAssignType,intAssignTrial,sprintf('%d ',vecPriorDistribution))
		%pause
	end
end

