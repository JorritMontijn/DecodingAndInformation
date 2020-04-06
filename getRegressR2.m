function [vecR2] = getRegressR2(matResp,vecNeuronsX,vecNeuronsY)
    %Returns array of reduced-rank R2 values from covariance matrix using
    %RGL's formula.
	
	%get submatrices
	matX = matResp(vecNeuronsX,:);
	matY = matResp(vecNeuronsY,:);
	intMaxRank = min([numel(vecNeuronsX) numel(vecNeuronsY)]);
	vecR2 = nan(1,intMaxRank);
	for intRank = 1:intMaxRank
		[matC, dblMSE, intRankOutT, sSuppOut] = doRdRankReg(matX', matY', 'rank', intRank);
		vecR2(intRank) = sSuppOut.dblR2;
	end
end
