clear
load('C:\Users\jacheung\Dropbox\LocalizationBehavior\DataStructs\BV.mat')
U=BV;
[V] = classifierWrapper_v2(BV,'pro','all'); %inputs = uberarray | touchDirection | touchOrder

%% PARAMETERS SETTING
clear trueXpreds
clear motorXpreds

% vars = {'countsBinary','counts'}; %Fig 3 %BUILD WITH ALL TOUCH DIRECTIONS AND NO drop
% vars = {'countsBinary','counts'}; %Fig 4 %BUILD WITH ALL TOUCH DIRECTIONS AND NO drop
% vars = {'kappa','timing','timeTotouch','counts','radialD','angle','uberedRadial'}; %FIG6
% vars = {'motor','kappa','timing','timeTotouch','counts','radialD','angle','uberedRadial'}; %Fig 7
% vars = {'uberedRadial'}; %for supplemental finding out optimal precision of model 

% Fig 9 and BEYOND PROTRACTION ONLY
% vars = {'angle','hilbert'} %Fig 7D
vars =  {'phase','amp','midpoint','angle'} %Fig9C
% vars = {'countsphase','countsamp','countsmidpoint','countsangle'}; %fig 9 D

% vars = {'counts','countsmidpoint','countsangle'};

savedLambdas = nan(length(V),length(vars));
%%
[rc] = numSubplots(length(vars));
for k = 1:length(vars)
    
    %designMatrix Parameters
    params.designvars = vars{k};
    % 1) 'angle' 2) 'hilbert' (phase amp midpoint) 3) 'counts' 4) 'ubered'
    % 5) 'timing' 6) 'motor' 7) 'decompTime' OR ,'kappa'
    % 'timeTotouch','onsetangle','velocity','Ivelocity' OR 'phase','amp','midpoint'
    params.classes = 'lick';
    % 1) 'gonogo' 2) 'lick'
    
    % Only for 'ubered' or 'hilbert'
    params.normalization = 'meanNorm';
    % 1) 'whiten'  2) 'meanNorm' 3)'none'
    
    params.dropNonTouch = 'yes';
    % 1) 'yes' = drop trials with 0 touches
    % 2) 'no' = keep all trials
    
    [DmatX, ~, ~] = designMatrixBuilder_v4(V(1),U{1},params);
    
    %learning parameters
    learnparam.regMethod = 'lasso';
    learnparam.lambda = loadLambda;
    % 1) 'lasso' or L1 regularization;
    % 2) 'ridge' or L2 regularization;
    learnparam.cvKfold = 5;
    learnparam.biasClose = 'no';
    learnparam.distance_round_pole =2; %inmm
    learnparam.numIterations = 20; 
    
    if size(DmatX,2)>1
        if sum(~isnan(savedLambdas(:,k)))==0
            [optLambda] = optimalLambda(V,U,params,learnparam);
            savedLambdas(:,k) = optLambda;
            learnparam.lambda = optLambda;
        else
            learnparam.lambda = savedLambdas(:,k);
        end
    else
        learnparam.lambda = zeros(1,length(U));
    end
    
    %% LOG CLASSIFIER
    
    accprop=cell(1,length(V));
    for rec = 1:length(V)
        
        clear opt_thresh
        motorPlick = [];
        motorPlickWithPreds = [];
        txp = [];
        
        for f = 1:learnparam.numIterations 
            
            display(['iteration ' num2str(f) ' for sample ' num2str(rec) ' using optimal lambda ' num2str(learnparam.lambda(rec))])
            [DmatX, DmatY, motorX] = designMatrixBuilder_v4(V(rec),U{rec},params); %lick/go = 1, nolick/nogo = 2; 
           
            if strcmp(learnparam.biasClose,'yes')
                mean_norm_motor = motorX - mean(BV{rec}.meta.ranges);
                close_trials = find(abs(mean_norm_motor)<learnparam.distance_round_pole*10000);
                DmatX = DmatX(close_trials,:);
                DmatY = DmatY(close_trials,:);
            end
            
            g1 = DmatY(DmatY == 1);
            g2 = DmatY(DmatY == 2);
            
            g1cvInd = crossvalind('kfold',length(g1),learnparam.cvKfold);
            g2cvInd = crossvalind('kfold',length(g2),learnparam.cvKfold);
            
            % shuffle permutations of cv indices
            g1cvInd = g1cvInd(randperm(length(g1cvInd)));
            g2cvInd = g2cvInd(randperm(length(g2cvInd)));
            
            selInds = [g1cvInd ;g2cvInd];
            
            for u=1:learnparam.cvKfold
                testY = [DmatY(selInds==u)];
                testX = [DmatX(selInds==u,:)];
                trainY = [DmatY(~(selInds==u))];
                trainX = [DmatX(~(selInds==u),:)];
                
                [beta,~,~] = ML_oneVsAll(trainX, trainY, numel(unique(DmatY)), learnparam.lambda(rec), learnparam.regMethod);
                weights.(vars{k}){rec}.theta{u}=beta;
                
                [pred,opt_thresh(u),prob]=ML_predictOneVsAll(beta,testX,testY,'Max');
                
                motorPlick= [motorPlick;motorX(selInds==u) prob(:,1)];
                motorPlickWithPreds= [motorPlickWithPreds ; motorX(selInds==u) prob pred];
                txp = [txp ; motorX(selInds==u) testY pred];
            end
        end  
            poptxp{rec} = txp;
            train_motorPlick_dual{rec} = motorPlickWithPreds; %used for predHeat
            train_predOpt(rec)=mean(opt_thresh); %used for dboundaries
    end
    
    %Raw truths and predictions: use this to calculate MCC, F1 scores,
    %etc...
    trueXpreds.(vars{k}) = poptxp;
    motorXpreds.(vars{k}) = train_motorPlick_dual;
    
    %Producing normalized odds ratios to compare weights between
    %features
    currWeight = weights.(vars{k});
    
    if size(DmatX,2)>1
        [nor] = ConvertToOddsRatio(currWeight,learnparam.cvKfold);
        
        figure(548);clf
        for e=1:size(nor,1)
            scatter(ones(1,length(currWeight))*e,nor(e,:),[],[.8 .8 .8],'filled')
            semnor = nanstd(nor(e,:))./sqrt(length(nor(e,:)));
            meannor = nanmean(nor(e,:));
            hold on; errorbar(e,meannor,semnor,'ko')      
        end
        
        for b = 1:size(nor,2)
            hold on;plot(1:size(nor,1),nor(:,b),'color',[.8 .8 .8])
        end
        
        
        hold on; plot(1:size(DmatX,2),mean(nor,2),'-k')
        hold on;plot([1 size(DmatX,2)],[0 0],'-.k')
        set(gca,'xlim',[0.75 size(nor,1)+.25],'ylim',[-1 1],'xtick',[1:size(nor,1)],'ytick',-1:.5:1)
        title(['norm odds for ' params.designvars])
        
        [p,~,stats]=anova1(nor',[],'off');
        pcomp = multcompare(stats,[],'off')
        
    end
    %
    
    
end


%% Psychometric Curve Comparison b/t Model and Mouse

    pfields = fields(motorXpreds);
    for d = 1:length(pfields)
        featureSelected = d;
        
        %addition of psycho with dropped values 
        for k = 1:length(U)
            all_motors = U{k}.meta.motorPosition; 
            curr_motors = motorXpreds.(pfields{featureSelected}){k}(:,1); 
            ntt_motors = setdiff(all_motors,curr_motors)'; 
            pad_values = [ntt_motors zeros(length(ntt_motors),1) ones(length(ntt_motors),1) ones(length(ntt_motors),1)*2]; %padding non-touch trials all as 0s meaning non-lick or nogo
            motorXpreds.(pfields{featureSelected}){k} = [motorXpreds.(pfields{featureSelected}){k} ; pad_values];
        end
            
        psycho = rebuildPsychoflip_v2(U,V,motorXpreds.(pfields{featureSelected}));
        suptitle([U{rec}.meta.layer ' ' pfields{featureSelected} ' ' params.classes])
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




%% Visualization of the decision boundaries.
colors = {'b','r'};
figure(12+k);clf
params.designvars = 'counts';
params.classes = 'gonogo';

params.normalization ='none';
for rec = 1:15
    [DmatX, DmatY, motorX, rnvals] = designMatrixBuilder_v4(V(rec),U{rec},params);
    nDimX = size(DmatX,2);
    switch nDimX
        case 1
            
%             x=linspace(min(DmatX),max(DmatX),15);
%  x = 0:0.0025:.05 %kappa
                x = 0:1:15 % counts
% x=0:25:750 %timing
%             x=0:4:60 %time to touch
%                 x = 0:2:40 %angle

            firstvar = histc(DmatX(DmatY==1),x);
            secondvar = histc(DmatX(DmatY==2),x);
            
            figure(12+k);subplot(3,5,rec)
            bar(x,firstvar/(sum(firstvar)+sum(secondvar)),colors{1});
            hold on;bar(x,secondvar/(sum(firstvar)+sum(secondvar)),colors{2});

            for db = [1:2]
                ms=cell2mat(weights.(params.designvars){rec}.theta);
                coords=mean(reshape(ms(db,:)',2,learnparam.cvKfold),2);
                y= (exp(coords(1)+coords(2)*x)) ./ (1 + exp(coords(1)+coords(2)*x))  ;
                y1=train_predOpt(rec);
                hold on; plot(x,y,['-.' colors{db}]);
                
%                 set(gca,'xlim',[min(x) max(x)],'ylim',[0 1]);

            end
%             alpha(.5)
            set(gca,'xlim',[min(x) max(x)],'ylim',[0 1]);

        
        case 2
            figure(12+k);subplot(3,5,rec)
            
            scatter(DmatX((DmatY==1),2),DmatX((DmatY==1),1),[],'filled','b')
            hold on;scatter(DmatX((DmatY==2),2),DmatX((DmatY==2),1),[],'filled','r')
%             alpha(.5)
            
            ms=cell2mat(weights.(params.designvars){rec}.theta);
            coords=mean(reshape(ms(1,:)',3,learnparam.cvKfold),2);
        
            plot_x = [min(DmatX(:,2)), max(DmatX(:,2))] ;
            plot_y = (-1./coords(2)) .* (coords(3).*plot_x + coords(1));
            hold on; plot(plot_x,plot_y,'-.k')
            
            set(gca,'xlim',[min(DmatX(:,2)) max(DmatX(:,2))],'ylim',[min(DmatX(:,1)) max(DmatX(:,1))]);
            
            title(num2str(mcc(rec,1)));
        case 3
            % %ASIDE: Plotting Decision Boundary for 3 variables
            
            ms=cell2mat(weights.(params.designvars){rec}.theta);
            coords=mean(reshape(ms(1,:)',4,learnparam.cvKfold),2);
            
            figure(10);clf
            scatter3(DmatX(DmatY==1,1),DmatX(DmatY==1,2),DmatX(DmatY==1,3),'m.')
            hold on;scatter3(DmatX(DmatY==2,1),DmatX(DmatY==2,2),DmatX(DmatY==2,3),'k.')
            
            %             hold on;scatter3(centroid(1,1),centroid(1,2),centroid(1,3),'k','linewidth',10)
            %             hold on;scatter3(centroid(2,1),centroid(2,2),centroid(2,3),'b','linewidth',10)
            
            plot_x = [min(DmatX(:,1))-2, min(DmatX(:,1))-2, max(DmatX(:,1))+2, max(DmatX(:,1))+2]; %ranges for amplitude
            plot_z = [-3 ,3,3,-3];
            plot_y = (-1/coords(3)) .* (coords(1) + (coords(2).*plot_x) + (coords(4).*plot_z) - log(train_predOpt(rec)/(1-train_predOpt(rec)))); % log p(go trial) to calculate decision boundary
            
            hold on; fill3(plot_x, plot_y, plot_z,'k');
            
            set(gca,'xlim',[min(DmatX(:,1)) max(DmatX(:,1))],'ylim',[min(DmatX(:,2)) max(DmatX(:,2))],'zlim',[min(DmatX(:,3)) max(DmatX(:,3))])
            if strcmp(params.designvars,'hilbert')
                xlabel('Amplitude');ylabel('Midpoint');zlabel('Phase')
            elseif strcmp(params.designvars,'decompTime')
                xlabel('time to touch');ylabel('whisk angle onset');zlabel('velocity')
            end
            
            
    end
    
end
suptitle(params.designvars)

print(['C:\Users\jacheung\Dropbox\LocalizationBehavior\Figures\Parts\' params.designvars params.classes],'-depsc')
