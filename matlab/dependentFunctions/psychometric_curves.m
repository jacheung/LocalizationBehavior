%% Psychometric Curve Comparison b/t Model and Mouse

    pfields = fields(mdl.output.motor_preds);
    for d = 1:length(pfields)
        featureSelected = d;
        
        %addition of psycho with dropped values 
        for k = 1:length(BV)
            all_motors = BV{k}.meta.motorPosition; 
            curr_motors = mdl.output.motor_preds.(pfields{featureSelected}){k}(:,1); 
            ntt_motors = setdiff(all_motors,curr_motors)'; 
            pad_values = [ntt_motors zeros(length(ntt_motors),1) ones(length(ntt_motors),1) ones(length(ntt_motors),1)*2]; %padding non-touch trials all as 0s meaning non-lick or nogo
            mdl.output.motor_preds.(pfields{featureSelected}){k} = [mdl.output.motor_preds.(pfields{featureSelected}){k} ; pad_values];
        end
            
        psycho = rebuildPsychoflip_v2(BV,V,mdl.output.motor_preds.(pfields{featureSelected}));
        suptitle([BV{rec}.meta.layer ' ' pfields{featureSelected} ' ' params.classes])
%         print(figure(5),'-depsc',['C:\Users\jacheung\Dropbox\LocalizationBehavior\Figures\currentBiologyUpdate\'  U{rec}.meta.layer '_' params.classes '_'  pfields{featureSelected} '_alltouches'])
        
        real = cellfun(@(x) cellfun(@(y) nanmean(y),x), psycho.mouse,'uniformoutput',0);
        mdled = cellfun(@(x) cellfun(@(y) nanmean(y),x), psycho.model,'uniformoutput',0);
        
        mae{d} = cellfun(@(x,y) mean(abs(x-y)),real,mdled);
       
        rsqd{d} = cellfun(@(x,y) corr(x,y).^2, real,mdled);
    end


x = repmat([1 2]',1,15);
figure(230);clf
% subplot(1,2,1);
% plot(cell2mat(rsqd(2:3)'),'ko-')
% set(gca,'xlim',[0.5 2.5],'xtick',[1 2],'xticklabel',{'mp+counts','angle+counts'},'ylim',[.7 1])
% ylabel('rsqd b/t model and mouse')
% [~,p] = ttest(rsqd{2},rsqd{3})
% title(['pairedTT p= ' num2str(p)])


plot(cell2mat(mae([1 3 2])'),'ko-')
set(gca,'xlim',[0.5 3.5],'xtick',[1:3],'xticklabel',{'counts','angle+counts','mp+counts'},'ylim',[0 .4],'ytick',0:.1:.4)
ylabel('mae b/t model and mouse')
[~,p] = ttest(mae{1},mae{2})
title(['pairedTT p= ' num2str(p)])
