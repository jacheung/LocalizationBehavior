clear
load('C:\Users\jacheung\Dropbox\LocalizationBehavior\DataStructs\BV.mat')
[V] = classifierWrapper_v2(BV,'all','all'); %inputs = uberarray | touchDirection | touchOrder

%% PARAMETERS SETTING
clearvars -except BV V

% vars = {'countsBinary','counts'}; %Fig 3 %BUILD WITH ALL TOUCH DIRECTIONS AND NO drop
vars = {'kappa','timing','timeTotouch','counts','radialD','angle'}; %FIG4G
% vars = {'motor','kappa','timing','timeTotouch','counts','radialD','angle','uberedRadial'}; %Fig 7
% vars = {'uberedRadial'}; %for supplemental finding out optimal precision of model 

% Fig 9 and BEYOND PROTRACTION ONLY
% vars = {'angle','hilbert'} %Fig 7D
% vars =  {'phase','amp','midpoint','angle'}; %Fig9C
% vars = {'countsphase','countsamp','countsmidpoint','countsangle'}; %fig 9 D

% vars = {'counts','countsmidpoint','countsangle'};
mdl = []; %starting w/ clean structure 
savedLambdas = nan(length(V),length(vars));
%%
[rc] = numSubplots(length(vars));
for k = 1:length(vars)
    
    %designMatrix Parameters
    params.designvars = vars{k};
    % 1) 'angle' 2) 'hilbert' (phase amp midpoint) 3) 'counts' 4) 'ubered'
    % 5) 'timing' 6) 'motor' 7) 'decompTime' OR ,'kappa'
    % 'timeTotouch','onsetangle','velocity','Ivelocity' OR 'phase','amp','midpoint'
    
    params.classes = 'gonogo';
    % 1) 'gonogo' 2) 'lick'
    
    % Only for 'ubered' or 'hilbert'
    params.normalization = 'meanNorm';
    % 1) 'whiten'  2) 'meanNorm' 3)'none'
    
    params.dropNonTouch = 'yes';
    % 1) 'yes' = drop trials with 0 touches
    % 2) 'no' = keep all trials
    
    [DmatX, ~, ~] = designMatrixBuilder_v4(V(1),BV{1},params);
    
    %learning parameters
    learnparam.regMethod = 'lasso'; % 'lasso' or L1 regularization or 'ridge' or L2 regularization;
    learnparam.lambda = loadLambda;

    learnparam.cvKfold = 5;
    learnparam.biasClose = 'no';
    learnparam.distance_round_pole =2; %inmm
    learnparam.numIterations = 20; 
    
    if size(DmatX,2)>1
        if sum(~isnan(savedLambdas(:,k)))==0
            [optLambda] = optimalLambda(V,BV,params,learnparam);
            savedLambdas(:,k) = optLambda;
            learnparam.lambda = optLambda;
        else
            learnparam.lambda = savedLambdas(:,k);
        end
    else
        learnparam.lambda = zeros(1,length(BV));
    end
    
    %% LOG CLASSIFIER
    
    for rec = 1:length(V)

        [DmatX, DmatY, motorX] = designMatrixBuilder_v4(V(rec),BV{rec},params); %lick/go = 1, nolick/nogo = 2;
        
        clear opt_thresh
        motorPlick = [];
        motorPlickWithPreds = [];
        txp = [];
        
        for f = 1:learnparam.numIterations  
            display(['iteration ' num2str(f) ' for sample ' num2str(rec) ' using optimal lambda ' num2str(learnparam.lambda(rec))])
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
            mdl.input.(vars{k}).DmatX{rec} = DmatX;
            mdl.input.(vars{k}).DmatY{rec} = DmatY;
            mdl.input.(vars{k}).motor{rec} = motorX; 
            
            mdl.output.true_preds.(vars{k}){rec} = txp;
            mdl.output.motor_preds.(vars{k}){rec} = motorPlickWithPreds;
            mdl.output.decision_boundary(rec) = mean(opt_thresh); %used for dboundaries
    end
    
    % Model Outputs; 
    mdl.build_params = params; %build parameters 
    mdl.learn_params = learnparam; %model learn parameters
    mdl.output.weights = weights; %model weights
    
end
mdl.build_params.designvars = vars; 


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




%% Visualization of the decision boundaries.
colors = {'b','r'};
figure(12+k);clf
params.designvars = 'radialD';
params.classes = 'gonogo';
params.normalization ='none';

for rec = 1:15
%     [DmatX, DmatY] = designMatrixBuilder_v4(V(rec),BV{rec},params); dont
%     need but can infact just use everything from mdl
    nDimX = size(DmatX,2);
    switch nDimX
        case 1
            if strcmp(params.designvars,'kappa')
                x = 0:0.0025:.05; %kappa
            elseif strcmp(params.designvars,'counts')
                x = 0:1:15; % counts
            elseif strcmp(params.designvars,'timing')
                x=0:25:750; %timing
            elseif strcmp(params.designvars,'timeTotouch')
                x=0:4:60; %time to touch
            elseif strcmp(params.designvars,'radialD')
                x = -5:.5:5; %radial distance NORMALIZED
            elseif strcmp(params.designvars,'angle')
                x = 0:2:40; %angle
            end

            firstvar = histc(DmatX(DmatY==1),x);
            secondvar = histc(DmatX(DmatY==2),x);
            
            figure(12+k);subplot(3,5,rec)
            bar(x,firstvar/(sum(firstvar)+sum(secondvar)),colors{1});
            hold on;bar(x,secondvar/(sum(firstvar)+sum(secondvar)),colors{2});

            for db = [1:2]
                ms=cell2mat(mdl.output.weights.(params.designvars){rec}.theta);
                coords=mean(reshape(ms(db,:)',2,learnparam.cvKfold),2);
                y= (exp(coords(1)+coords(2)*x)) ./ (1 + exp(coords(1)+coords(2)*x))  ;
                hold on; plot(x,y,['-.' colors{db}]);
            end
            
            alpha(.5)
            set(gca,'xlim',[min(x) max(x)],'ylim',[0 1]);

        
        case 2
            figure(12+k);subplot(3,5,rec)
            
            scatter(DmatX((DmatY==1),2),DmatX((DmatY==1),1),[],'filled','b')
            hold on;scatter(DmatX((DmatY==2),2),DmatX((DmatY==2),1),[],'filled','r')
%             alpha(.5)
            
            ms=cell2mat(mdl.output.weights.(params.designvars){rec}.theta);
            coords=mean(reshape(ms(1,:)',3,learnparam.cvKfold),2);
        
            plot_x = [min(DmatX(:,2)), max(DmatX(:,2))] ;
            plot_y = (-1./coords(2)) .* (coords(3).*plot_x + coords(1));
            hold on; plot(plot_x,plot_y,'-.k')
            
            set(gca,'xlim',[min(DmatX(:,2)) max(DmatX(:,2))],'ylim',[min(DmatX(:,1)) max(DmatX(:,1))]);
            
            title(num2str(mcc(rec,1)));
        case 3
            % %ASIDE: Plotting Decision Boundary for 3 variables
            
            ms=cell2mat(mdl.output.weights.(params.designvars){rec}.theta);
            coords=mean(reshape(ms(1,:)',4,learnparam.cvKfold),2);
            
            figure(10);clf
            scatter3(DmatX(DmatY==1,1),DmatX(DmatY==1,2),DmatX(DmatY==1,3),'m.')
            hold on;scatter3(DmatX(DmatY==2,1),DmatX(DmatY==2,2),DmatX(DmatY==2,3),'k.')
            
            %             hold on;scatter3(centroid(1,1),centroid(1,2),centroid(1,3),'k','linewidth',10)
            %             hold on;scatter3(centroid(2,1),centroid(2,2),centroid(2,3),'b','linewidth',10)
            
            plot_x = [min(DmatX(:,1))-2, min(DmatX(:,1))-2, max(DmatX(:,1))+2, max(DmatX(:,1))+2]; %ranges for amplitude
            plot_z = [-3 ,3,3,-3];
            plot_y = (-1/coords(3)) .* (coords(1) + (coords(2).*plot_x) + (coords(4).*plot_z) - log(mdl.output.decision_boundary(rec)/(1-mdl.output.decision_boundary(rec)))); % log p(go trial) to calculate decision boundary
            
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
