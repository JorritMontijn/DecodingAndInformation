function [matY_hat,dblR2_CV,matB] = doCrossValidatedDecodingRR(matX,matY,varTypeCV,dblLambda)
	%doCrossValidatedDecodingRR Cross-validated linear ridge regression decoder
	%[matY_hat,matB] = doCrossValidatedDecodingRR(matX,matY,varTypeCV,dblLambda)
	%
	%Inputs:
	% - matX; [n x p] Matrix of n observations/trials of p predictors/neurons
	% - matY; [n x q] Matrix of n observations/trials of q variables
	% - varTypeCV; [int or vec] Integer switch 0-1 or n-element [train=0/test=1] vector. 
	%				Val=0, no CV; val=1, leave-one-out CV, vector=test/train set
	% - dblLambda; [scalar] Ridge regularization parameter 
	%
	%Outputs:
	% - matY_hat; Predicted values
	% - dblR2_CV; R^2 of Y_hat
	% - matB; [p x q] Regression matrix
	%
	%Version History:
	%2021-02-22 Created function [by Jorrit Montijn]
	
	%% check inputs
	if ~exist('varTypeCV','var') || isempty(varTypeCV)
		varTypeCV = 1;
	end
	if ~exist('dblLambda','var') || isempty(dblLambda)
		dblLambda = 1;
	end
	[n,p] = size(matX);
	[n,q] = size(matY);
	if size(matX,1) ~= size(matY,1)
		error([mfilename ':ObservationNumberMismatch'],sprintf('Incongruous number of observations: X has %d, Y has %d',size(matX,1),size(matY,1)));
	end
	if numel(varTypeCV) > 1 && size(matX,1) ~= numel(varTypeCV)
		error([mfilename ':ObservationNumberMismatch'],sprintf('Incongruous number of observations: CV-vector has %d, X has %d',numel(varTypeCV),size(matX,1)));
	end
	%check if input is zero-centered
	dblWarnLim =  1e-10;
	dblMaxMean = max(abs(mean(matX,2) ./ range(matX,2)));
	if dblMaxMean > dblWarnLim
		warning([mfilename ':MeansNotZeroCentered'],'Means in X are not zero-centered; this probably reduces linear predictability!');
	end
	
	%% which CV?
	matTestY = matY;
	if numel(varTypeCV)>1
		%% divide test/train data
		indTrain = varTypeCV == 0;
		indTest = varTypeCV == 1;
		matTrainX = matX(indTrain,:);
		matTestX = matX(indTest,:);
		matTrainY = matY(indTrain,:);
		matTestY = matY(indTest,:);
		intPredictors = size(matTrainX,2);
		
		%% perform ridge regression
		matB = (matTrainX' * matTrainX + dblLambda*eye(intPredictors)) \ (matTrainX' * matTrainY); %left-divide is same as inverse and multiplication
		
		%% compute test performance
		%predict responses
		matY_hat = matTestX * matB;
	elseif varTypeCV==0
		%% divide test/train data
		intPredictors = size(matX,2);
		
		%% perform ridge regression
		matB = (matX' * matX + dblLambda*eye(intPredictors)) \ (matX' * matY); %left-divide is same as inverse and multiplication
		
		%% compute test performance
		%predict responses
		matY_hat = matX * matB;
	elseif varTypeCV==1
		%% pre-allocate
		matB = zeros(p,q); %[p x q]
		matY_hat = nan(n,q); %[n x q]
		
		for intObs=1:n
			%% divide test/train data
			indTrain = true(n,1);
			indTrain(intObs) = false;
			indTest = ~indTrain;
			matTrainX = matX(indTrain,:);
			matTestX = matX(indTest,:);
			matTrainY = matY(indTrain,:);
			intPredictors = size(matTrainX,2);
			
			%% perform ridge regression
			matB1 = (matTrainX' * matTrainX + dblLambda*eye(intPredictors)) \ (matTrainX' * matTrainY); %left-divide is same as inverse and multiplication
			matB = (matB*(intObs-1) + matB1)/intObs; %iterative construction of matB
		
			%% compute test performance
			%predict responses
			matY_hat(intObs,:) = matTestX * matB1;
		end
		%matY_hat = matX * matB;
	else
		error([mfilename ':WrongCVtype'],'CV type not recognized');
	end
	
	%% calculate summary statistics
	%get R^2
	vecMu = mean(matTestY);
	dblSSRes_ridge = sum(sum((matTestY - matY_hat).^2));
	dblSSTot = sum(sum(bsxfun(@minus,matTestY,vecMu).^2));
	dblR2_CV = 1 - dblSSRes_ridge / dblSSTot;

	%{
	% compute train performance for sanity checks
	%predict responses
	matY_pred_Train = matTrainX * matB;

	%get R^2
	vecMu = mean(matTrainY);
	dblSSRes_Train = sum(sum((matTrainY - matY_pred_Train).^2));
	dblSSTot_Train = sum(sum(bsxfun(@minus,matTrainY,vecMu).^2));
	dblR2_NonCV = 1 - dblSSRes_Train / dblSSTot_Train;
	%}
end



