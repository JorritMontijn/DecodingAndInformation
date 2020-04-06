function [vecFisherProj, vecFisherOrth, vecNoisePred] = doFisherAndPredConsecutiveSubspaces2(vecFprime, matCov, intMaxRemDim,matX, matY)
	% Calculates Fisher information in projection space.
	% columns of matW should form basis vectors for projection space
	% vecFprime should be column vector
	
	vecFisherProj = zeros(1,intMaxRemDim);
	vecFisherOrth = zeros(1,intMaxRemDim);
	vecNoisePred = zeros(1,intMaxRemDim);
	
	%prep
	intT = size(matX, 1); %trials
	intN = size(matX, 2); %neurons
	vecMu = mean(matY);
	 
    matMuX = repmat(mean(matX), [size(matX,1) 1]);
    matMuY = repmat(vecMu, [size(matY,1) 1]);
    matX_fluc = matX - matMuX;
    matY_fluc = matY - matMuY;
	cellMatY = [];
	cellMatY{1} = matY_fluc;
    matLatentData = nan(intT,intMaxRemDim);
    matP_orth = eye(intN);
    matW_orth = eye(intN);
    
	for intRemDim = 1:intMaxRemDim
		vecW = matCov\vecFprime;
		[vecFisherProj(intRemDim), vecFisherOrth(intRemDim)] = getFisherInSubspace(vecFprime, matCov, vecW);
        
        vecW = matW_orth*vecW;
        matCov = matW_orth*matCov*matW_orth';
        vecFprime = matW_orth*vecFprime;

		%% noise pred
		% get prediction
        matLatentData(:,intRemDim) = matX_fluc*vecW;
		cellMatX = [];
		cellMatX{1} = matLatentData(:,1:intRemDim);
		vecNoisePred(intRemDim) = doDimPredFull(cellMatX,cellMatY,1);
		 
		%get orthogonal space
		matP_orth = matP_orth - vecW*vecW'/(vecW'*vecW);
        matW_orth = orth(matP_orth);
		matP_latent_orth = (matW_orth'*matW_orth)\matW_orth';
		vecFprime = matP_latent_orth*vecFprime;
		matCov = matP_latent_orth*matCov*matP_latent_orth';
	end
	
end
