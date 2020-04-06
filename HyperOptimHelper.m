function vecOut = HyperOptimHelper(vecOptimP,matX)
	%UNTITLED2 Summary of this function goes here
	%   Detailed explanation goes here
	global HyperOptim_cellVarIn;
	global HyperOptim_strSolver;
	global HyperOptim_fFunction;
	global HyperOptim_vecOptimParamsIdx;
	global strEval;
	
	%build inputs
	intVarNum = max([numel(HyperOptim_cellVarIn) HyperOptim_vecOptimParamsIdx]);
	
	%% run
	if strcmpi(HyperOptim_strSolver,'fminsearch') || strcmpi(HyperOptim_strSolver,'fminbnd')
		strEval = strcat(HyperOptim_fFunction,'(');
		intStartIn = 1;
	else
		strEval = strcat(HyperOptim_fFunction,'(matX,');
		intStartIn = 2;
	end
	for intVar=intStartIn:intVarNum
		%assign arguments
		intOptimIdx = find(HyperOptim_vecOptimParamsIdx==intVar);
		if ~isempty(intOptimIdx)
			strOptimIdx = num2str(intOptimIdx);
			strEval = strcat(strEval,'vecOptimP(',strOptimIdx,'),');
		else
			strEval = strcat(strEval,'HyperOptim_cellVarIn{',num2str(intVar),'},');
		end
	end
	strEval(end:end+1) = ');';
	%get value
	%vecOptimP
	vecOut = eval(strEval);
end

