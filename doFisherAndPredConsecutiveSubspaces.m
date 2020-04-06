function [vecFisherProj, vecFisherOrth, vecNoisePred] = doFisherAndPredConsecutiveSubspaces(vecFprime, matCov, intMaxRemDim,matX, matY)
	% Calculates Fisher information in projection space.
	% columns of matW should form basis vectors for projection space
	% vecFprime should be column vector
	
	vecFisherProj = zeros(1,intMaxRemDim);
	vecFisherOrth = zeros(1,intMaxRemDim);
	vecNoisePred = zeros(1,intMaxRemDim);
	
	%prep
	intN = size(matX, 1); %samples (trials)
	vecMu = mean(matY);
	dblSSTot = sum(sum(bsxfun(@minus,matY,vecMu).^2));
    
    matMuX = repmat(mean(matX), [size(matX,1) 1]);
    matMuY = repmat(vecMu, [size(matY,1) 1]);
    matX_fluc = matX - matMuX;
    matY_fluc = matY - matMuY;
	
	for intRemDim = 1:intMaxRemDim
		vecW = matCov\vecFprime;
		[vecFisherProj(intRemDim), vecFisherOrth(intRemDim)] = getFisherInSubspace(vecFprime, matCov, vecW);
		
		%% noise pred
		% get prediction
		matW = repmat(vecW, [1 size(matY,2)]);
        matY_fluc_pred = matX_fluc*matW;  %predicted (matY-meanY) up to a scaling (to be computed in next line)
        alpha = trace(matY_fluc*matY_fluc_pred')/trace(matY_fluc_pred*matY_fluc_pred');
        matY_pred = alpha*matY_fluc_pred + matMuY;  %predicted matY
		
		% compute MSE
		matErr = (matY - matY_pred).^2;
		%get R^2
		dblSSRes = sum(sum(matErr));
		vecNoisePred(intRemDim) = 1 - dblSSRes / dblSSTot;
		
		
		%get orthogonal space
		matW_orth = orth(eye(numel(vecFprime)) - vecW*((vecW'*vecW)\vecW'));
		matP_orth = (matW_orth'*matW_orth)\matW_orth';
		vecFprime = matP_orth*vecFprime;
		matX_fluc = matX_fluc*matP_orth';
		
		matCov = matP_orth*matCov*matP_orth';
	end
	
end
