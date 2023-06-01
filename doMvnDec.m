function matTestPosterior = doMvnDec(matTrainData,vecTrainTrialType,matTestData,dblLambda)
	%% calculate test probabilities by fitting a multivariate gaussian to the training data
	%get variables
	matSampleData = [matTestData matTrainData];
	[intNeurons,intTrials] = size(matSampleData);
	intStimTypes = length(unique(vecTrainTrialType));
	intTrainTrials = size(matTrainData,2);
	intTestTrials = size(matTestData,2);
	
	%prep data
	matMeans = NaN(intNeurons,intStimTypes);
	for k = 1:intStimTypes
		matMeans(:,k) = mean(matTrainData(:,vecTrainTrialType==k),2);
	end
	
	%center data
	matTrainCentered = matTrainData' - matMeans(:,vecTrainTrialType)';
	
	% QR decomposition
	X = qr(matTrainCentered, 0);
	R = triu(X);
	R = R / sqrt(intTrainTrials - intStimTypes); % SigmaHat = R'*R
	s = svd(R);
	logDetSigma = 2*sum(log(s)); % avoid over/underflow
	
	% calculate log probabilities
	D_full = NaN(intTrials, intStimTypes,'single');
	for k = 1:intStimTypes
		A = bsxfun(@minus,matSampleData', matMeans(:,k)') / R;
		D_full(:,k) = log(1/intStimTypes) - .5*(sum(A .* A, 2) + logDetSigma);
	end
	
	if dblLambda > 0 %skip if not required
		%naive Bayes (independent)
		S = std(matTrainCentered) * sqrt((intTrainTrials-1)./(intTrainTrials-intStimTypes));
		D_diag = NaN(intTrials, intStimTypes);
		for k = 1:intStimTypes
			A=bsxfun(@times, bsxfun(@minus,matSampleData',matMeans(:,k)'),1./S);
			D_diag(:,k) = log(1/intStimTypes) - .5*(sum(A .* A, 2) + logDetSigma);
		end
	else
		D_diag = 0;
	end
	
	
	if isinf(dblLambda)%special case to avoid numerial overflow
		%take only D_diag
		D = D_diag;
	else
		%weight probabilities by lambda ratio
		D = (dblLambda*D_diag + D_full) / (1 + dblLambda);
	end
	
	% find highest log probability for each trial
	maxD = max(D, [], 2);
	
	%because of earlier reordering, the first intTestTrials trials are the test set
	% Bayes' rule: first compute p{x,G_j} = p{x|G_j}Pr{G_j} ...
	% (scaled by max(p{x,G_j}) to avoid over/underflow)
	% ... then Pr{G_j|x) = p(x,G_j} / sum(p(x,G_j}) ...
	% (numer and denom are both scaled, so it cancels out)
	
	%likelihoods of test data for each class, scaled to max likelihood
	P = exp(bsxfun(@minus,D(1:intTestTrials,:),maxD(1:intTestTrials)));
	%rescale over P
	sumP = nansum(P,2);
	
	%assign output
	matTestPosterior = bsxfun(@times,P,1./(sumP));
end