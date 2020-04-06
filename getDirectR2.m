function [vecR2,vecEigVals] = getDirectR2(matCovariance,vecNeuronsX,vecNeuronsY)
    %Returns array of reduced-rank R2 values from covariance matrix using
    %RGL's formula.
    
	%get submatrices
    matXX = matCovariance(vecNeuronsX,vecNeuronsX);
    matXY = matCovariance(vecNeuronsX,vecNeuronsY);
    matYX = matCovariance(vecNeuronsY,vecNeuronsX);
    matYY = matCovariance(vecNeuronsY,vecNeuronsY);
    
	%calc R^2 using Rex's technique
    vecEigVals = eig(matYX * (matXX \ matXY));
    vecEigVals = sort(vecEigVals, 'descend');
    vecR2 = cumsum(vecEigVals)./trace(matYY);
end
