function [dblFisherProj, dblFisherOrth, dblFisherTotal] = getFisherInSubspace(vecFprime, matCov, matW)
	% Calculates Fisher information in projection space.
	% columns of matW should form basis vectors for projection space
	% vecFprime should be column vector
	
	%% version 1
	%{
	%calculate projection matrix
	matP = (matW'*matW)\matW';
	
	%extract orthogonal basis
	matW_orth = orth(eye(numel(vecFprime)) - matW*matP);
	
	%calculate orthogonal projection matrix
	matP_orth = (matW_orth'*matW_orth)\matW_orth';
	
	%get information in projected space
	dblFisherProj = vecFprime'*matP'/(matP*matCov*matP')*matP*vecFprime;
	
	%get information in null-space
	dblFisherOrth = vecFprime'*matP_orth'/(matP_orth*matCov*matP_orth')*matP_orth*vecFprime;
	end
	toc
	
	if nargout > 2
		%get total information
		dblFisherTotal = vecFprime'/matCov*vecFprime;
	end
	%}
	%% version 2
	if nargin < 3 || isempty(matW)
		dblFisherProj = vecFprime'/matCov*vecFprime;
	else
		%get information in projected space
		dblFisherProj = vecFprime'*matW/(matW'*matCov*matW)*matW'*vecFprime;
	end
	
	if nargout > 1
		%calculate projection matrix
		matP = (matW'*matW)\matW';
		
		%extract orthogonal basis
		matW_orth = orth(eye(numel(vecFprime)) - matW*matP);
		
		%get information in null-space
		dblFisherOrth = vecFprime'*matW_orth/(matW_orth'*matCov*matW_orth)*matW_orth'*vecFprime;
	end
	if nargout > 2
		%get total information
		dblFisherTotal = vecFprime'/matCov*vecFprime;
	end
end
