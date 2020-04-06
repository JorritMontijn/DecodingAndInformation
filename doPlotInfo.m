function doPlotInfo(vecGroupSizes,matI_Dir,matI_Dir_shuff,matA_ML,matA_ML_shuff)
	%UNTITLED Summary of this function goes here
	%   Detailed explanation goes here
	
	%% get vars
	intGroupSizes = numel(vecGroupSizes);
	
	%% plot
	vecMeanI = nan(1,intGroupSizes);
	vecMeanI_shuff = nan(1,intGroupSizes);
	vecMeanA = nan(1,intGroupSizes);
	vecMeanA_shuff = nan(1,intGroupSizes);
	vecSDI = nan(1,intGroupSizes);
	vecSDI_shuff = nan(1,intGroupSizes);
	vecSDA = nan(1,intGroupSizes);
	vecSDA_shuff = nan(1,intGroupSizes);
	for intGroupSize=1:intGroupSizes
		matValsI = matI_Dir(:,:,:,intGroupSize);
		vecMeanI(intGroupSize) = nanmean(matValsI(:));
		vecSDI(intGroupSize) = nanstd(matValsI(:));
	
		matValsI_shuff = matI_Dir_shuff(:,:,:,intGroupSize);
		vecMeanI_shuff(intGroupSize) = nanmean(matValsI_shuff(:));
		vecSDI_shuff(intGroupSize) = nanstd(matValsI_shuff(:));
		
		matValsA = matA_ML(:,:,:,intGroupSize);
		vecMeanA(intGroupSize) = nanmean(matValsA(:));
		vecSDA(intGroupSize) = nanstd(matValsA(:));
		
		matValsA_shuff = matA_ML_shuff(:,:,:,intGroupSize);
		vecMeanA_shuff(intGroupSize) = nanmean(matValsA_shuff(:));
		vecSDA_shuff(intGroupSize) = nanstd(matValsA_shuff(:));
	end
	
	%plot
	clf
	subplot(2,2,1)
	hold on
	errorbar(vecGroupSizes,vecMeanI,vecSDI,'r-');
	errorbar(vecGroupSizes,vecMeanI_shuff,vecSDI_shuff,'r--');
	hold off
	title('Fisher information; dotted=shuff')
	xlim([0 max(get(gca,'xlim'))]);ylim([0 max(get(gca,'ylim'))]);
	xlabel('Population size');
	ylabel('Fisher Information')
	fixfig;

	
	subplot(2,2,2)
	hold on
	errorbar(vecGroupSizes,vecMeanA,vecSDA,'b-');
	errorbar(vecGroupSizes,vecMeanA_shuff,vecSDA_shuff,'b--');
	hold off
	title('ML decoding; dotted=shuff')
	ylim([min(get(gca,'ylim')) 1]);xlim([0 max(get(gca,'xlim'))]);
	xlabel('Population size');
	ylabel('Accuracy')
	fixfig;
	
	%diagonal only
	%% plot
	vecMeanI = nan(1,intGroupSizes);
	vecMeanI_shuff = nan(1,intGroupSizes);
	vecMeanA = nan(1,intGroupSizes);
	vecMeanA_shuff = nan(1,intGroupSizes);
	vecSDI = nan(1,intGroupSizes);
	vecSDI_shuff = nan(1,intGroupSizes);
	vecSDA = nan(1,intGroupSizes);
	vecSDA_shuff = nan(1,intGroupSizes);
	for intGroupSize=1:intGroupSizes
		matValsI = matI_Dir(:,:,:,intGroupSize);
		vecMeanI(intGroupSize) = nanmean(diag(matValsI,-1));
		vecSDI(intGroupSize) = nanstd(diag(matValsI,-1));
	
		matValsI_shuff = matI_Dir_shuff(:,:,:,intGroupSize);
		vecMeanI_shuff(intGroupSize) = nanmean(diag(matValsI_shuff,-1));
		vecSDI_shuff(intGroupSize) = nanstd(diag(matValsI_shuff,-1));
		
		matValsA = matA_ML(:,:,:,intGroupSize);
		vecMeanA(intGroupSize) = nanmean(diag(matValsA,-1));
		vecSDA(intGroupSize) = nanstd(diag(matValsA,-1));
		
		matValsA_shuff = matA_ML_shuff(:,:,:,intGroupSize);
		vecMeanA_shuff(intGroupSize) = nanmean(diag(matValsA_shuff,-1));
		vecSDA_shuff(intGroupSize) = nanstd(diag(matValsA_shuff,-1));
	end
	
	%plot
	subplot(2,2,3)
	hold on
	errorbar(vecGroupSizes,vecMeanI,vecSDI,'r-');
	errorbar(vecGroupSizes,vecMeanI_shuff,vecSDI_shuff,'r--');
	hold off
	title('Fisher information adjacent; dotted=shuff')
	xlim([0 max(get(gca,'xlim'))]);ylim([0 max(get(gca,'ylim'))]);
	xlabel('Population size');
	ylabel('Fisher Information')
	fixfig;
	
	subplot(2,2,4)
	hold on
	errorbar(vecGroupSizes,vecMeanA,vecSDA,'b-');
	errorbar(vecGroupSizes,vecMeanA_shuff,vecSDA_shuff,'b--');
	hold off
	title('ML decoding adjacent; dotted=shuff')
	ylim([min(get(gca,'ylim')) 1]);xlim([0 max(get(gca,'xlim'))]);
	xlabel('Population size');
	ylabel('Accuracy')
	fixfig;
	
	drawnow;
end

