function plotRxnTime(array)   
%reaction time is defined as the mean time from first touch to first lick
%after first touch
for i  = 1:length(array)
pOnset = round(array{i}.meta.poleOnset(1)*1000);
        for k = 1:array{i}.k
            touch(k) = min([find(array{i}.S_ctk(9,pOnset:4000,k)==1,1) 4000])+pOnset;
            lix(k) =min([find(array{i}.S_ctk(16,touch(k):4000,k)==1,1) 4000])+touch(k);
        end
%         lix(lix==4000+pOnset+samplingPeriod)=nan;
        touch(touch==4000+pOnset)=nan;
        lix(lix>=4000)=nan;
        rxntimetmp = lix-touch;

        rxntime(i) = nanmean(rxntimetmp(rxntimetmp>0));
end

%RXN TIME
figure(580);clf
scatter(rxntime,ones(1,length(array)),'markerfacecolor',[.8 .8 .8],'markeredgecolor',[.8 .8 .8]);
hold on; errorbar(mean(rxntime),1,std(rxntime),'horizontal','ko','markerfacecolor','k','markeredgecolor','k','markersize',20)
set(gca,'ylim',[.5 1.5],'ytick',[],'xtick',0:250:1250,'xlim',[0 1250])
xlabel('reaction time (ms)')

set(gcf, 'Units', 'pixels', 'Position', [250, 250, 500, 200]);