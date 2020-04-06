function [vecFisherProj, vecFisherOrth] = doFisherConsecutiveSubspaces(vecFprime, matCov, intMaxRemDim)
	% Calculates Fisher information in projection space.
	% columns of matW should form basis vectors for projection space
	% vecFprime should be column vector
	
	vecFisherProj = zeros(1,intMaxRemDim);
	vecFisherOrth = zeros(1,intMaxRemDim);
	
	for intRemDim = 1:intMaxRemDim
		matW = matCov\vecFprime;
		[vecFisherProj(intRemDim), vecFisherOrth(intRemDim)] = getFisherInSubspace(vecFprime, matCov, matW);
		
		matW_orth = orth(eye(numel(vecFprime)) - matW*((matW'*matW)\matW'));
		matP_orth = (matW_orth'*matW_orth)\matW_orth';
		vecFprime = matP_orth*vecFprime;
		matCov = matP_orth*matCov*matP_orth';
	end
	
end
