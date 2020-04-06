function vecOptimParamsVals = HyperOptim(fFunction,vecOptimParamsIdx,cellVarIn,vecStartVals,vecLowerB,vecUpperB,strSolver,matX,vecY)
	%HyperDecoder Optimize hyperparameter. Syntax:
	%   dblParam = HyperOptim('function',vecOptimParams,{varIn1,varIn2,...},vecStartVals,vecLowerB,vecUpperB,strSolver)
	%
	%Example:
	%dblOptimLambda = HyperOptim('doCrossValidatedDecodingLR',4,{matAgg,vecAggTrialTypes,1},10000,1,1e9,'fminbnd');

	%check inputs
	if ~exist('strSolver','var') || isempty(strSolver)
		strSolver = 'fminbnd';
	end
	
	%create globals & send extra vars to helper
	global HyperOptim_cellVarIn;
	global HyperOptim_strSolver;
	global HyperOptim_fFunction;
	global HyperOptim_vecOptimParamsIdx;
	HyperOptim_cellVarIn = cellVarIn;
	HyperOptim_strSolver = strSolver;
	HyperOptim_fFunction = fFunction;
	HyperOptim_vecOptimParamsIdx = vecOptimParamsIdx;
	
	if strcmpi(strSolver,'lsqcurvefit')
		%vecLinCoeffs = lsqcurvefit('gnmlinfunc', vecLinCoeffs0, matX, vecY,vecLinCoeffsLB,vecLinCoeffsUB,sOptimOptions);
		vecOptimParamsVals = curvefitfun('HyperOptimHelper', vecStartVals, matX, vecY,vecLowerB,vecUpperB);
	elseif strcmpi(strSolver,'lsqnonlin')
		vecOptimParamsVals = lsqnonlin('HyperOptimHelper', vecStartVals,vecLinCoeffsLB,vecLinCoeffsUB);
	elseif strcmpi(strSolver,'fminsearch')
		vecOptimParamsVals = fminsearch(@HyperOptimHelper,vecStartVals);
	elseif strcmpi(strSolver,'fminbnd')
		vecOptimParamsVals = fminbnd(@HyperOptimHelper,vecLowerB,vecUpperB);
	end
	
end

