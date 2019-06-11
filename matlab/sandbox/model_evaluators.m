% model evaluators for seeing :1) distance of prediction from mouse and :2)
% model accuracy resolution
%% Choice from optimal prediction  %requires U,V,trueXpreds

pfields = fields(trueXpreds);
for k = 1:length(pfields)
    cvs = trueXpreds.(pfields{k});
    
    for rec = 1:length(U)
    real=[U{rec}.meta.motorPosition;V(rec).trialNums.matrix(5,:)]';%only taking lick row and motorPos row
    umotors = unique(cvs{rec}(:,1)) ;
    accuracy = cell(numel(umotors),1); 
    for d = 1:length(umotors)
        optIdx = find(cvs{rec}(:,1) == umotors(d));
        optimal = cvs{rec}(optIdx,3);
        optimal(optimal==2)=0; %recode optimal values so it'll be for nogo
        choiceIdx = find(real(:,1)==umotors(d)); 
        choice = real(choiceIdx,2); 
        accuracy{d} = optimal==choice(1);
    end
    
    ntts = setdiff(real(:,1),umotors);
    for g = 1:length(ntts)
        optimal = 0; %set as nogo 
        choiceIdx = find(real(:,1)==ntts(g));
        choice = real(choiceIdx,2); 
        accuracy{d+g} = optimal == choice(1);
    end
    allMotors = [umotors ; ntts]; 
    [realsorted]= binslin(allMotors,cellfun(@nanmean,accuracy),'equalE',13,min(allMotors),max(allMotors));     
    acc.(pfields{k}){rec} = cellfun(@nanmean ,realsorted);
    end
end
     
figure(213);clf
for k = 1:15
    subplot(3,5,k);
    x=linspace(-1,1,numel(acc.counts{k}));
    plot(x,flipud(acc.counts{k}),'color',[.8 .8 .8]);
    hold on; plot(x,flipud(acc.countsmidpoint{k}),'color','g');
    hold on; plot(x,flipud(acc.countsangle{k}),'color','b');
    set(gca,'xlim',[-1 1],'ylim',[0 1])
end
    
%% Resolution of model from optimal prediction  %requires U,V,trueXpreds  
pfields = fields(trueXpreds);
color = {'k','g','b'};
figure(32);clf

for k = 2:length(pfields)
    cvs = trueXpreds.(pfields{k});
    binned = cellfun(@(x) binslin(x(:,1),x(:,[2 3]),'equalE',11,min(x(:,1)),max(x(:,1))),cvs,'uniformoutput',0);
    pc = cellfun(@(x) cellfun(@(y) mean(y(:,1) == y(:,2)),x),binned,'uniformoutput',0);
    pcFA = cellfun(@(x) [ones(length(x)./2,1) - x(1:length(x)./2) ; x((length(x)./2)+1:end)] ,pc,'uniformoutput',0) ; %convert to FArate
    
    binnedFine = cellfun(@(x) binslin(x(:,1),x(:,[2 3]),'equalE',21,min(x(:,1)),max(x(:,1))),cvs,'uniformoutput',0);
    pcFine = cellfun(@(x) cellfun(@(y) mean(y(:,1) == y(:,2)),x),binnedFine,'uniformoutput',0);
    pcFAFine = cellfun(@(x) [ones(length(x)./2,1) - x(1:length(x)./2) ; x((length(x)./2)+1:end)] ,pcFine,'uniformoutput',0) ; %convert to FArate
    
    
     ts = tinv(0.975,numel(binned)-1);      % T-Score @ 95% for calculating 95CI
    
    finemnsct = nanmean(cell2mat(pcFAFine),2);
    finesemsct = nanstd(cell2mat(pcFAFine),[],2) ./ sqrt(numel(binned)); %SEM error
    fineci = abs(ts.*finesemsct);  %95CI error
    finerr = fineci; 

    fasfine = [finemnsct(1:length(finemnsct)./2) finerr(1:length(finerr)./2)];
    hitsfine = flipud([finemnsct((length(finemnsct)./2)+1:end) finerr((length(finerr)./2)+1:end)]);
    
    mnsct = nanmean(cell2mat(pcFA),2);
    semsct = nanstd(cell2mat(pcFA),[],2) ./ sqrt(numel(binned)); %SEM error
    ci = abs(ts.*semsct); %95CI error
    error = ci; %error chosen 
    
    fas = [mnsct(1:length(mnsct)./2) error(1:length(error)./2)];
    hits = flipud([mnsct((length(mnsct)./2)+1:end) error((length(error)./2)+1:end)]);
    
    gfas = [fas ; fasfine(end,:)];
    ghits = [hits ; hitsfine(end,:)];
    
    
    hold on; errorbar(gfas(:,1),ghits(:,1),ghits(:,2),ghits(:,2),gfas(:,2),gfas(:,2),[color{k} '-o']);
    
    
%     subplot(1,3,k)
%     for b = 1:length(gfas)
%     hold on; errorbar(gfas(b,1),ghits(b,1),ghits(b,2),ghits(b,2),gfas(b,2),gfas(b,2),'o','color',color(b,:));
%     end
    hold on; plot([0 1],[0 1],'-.k')
    set(gca,'xtick',[0 1],'ytick',[0 1],'ylim',[0 1],'xlim',[0 1]);
    axis square
    xlabel('FA rate');ylabel('hit rate')
%     title(pfields{k})
    
%     if k == length(k)
%         legend('5mm','4mm','3mm','2mm','1mm','.5mm')
%     end
end
    
[outputs] = discrimination_precision(U);
figure(32); 
hold on; errorbar(outputs.means(:,1),outputs.means(:,2),outputs.errors(:,2),outputs.errors(:,2),outputs.errors(:,1),outputs.errors(:,1),'k-o')

    
    