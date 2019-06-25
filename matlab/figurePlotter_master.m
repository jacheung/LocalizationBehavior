%%__main__
close all; clear; clc; 
%% Load behavioral data matrix 
behavioralDataLocation = 'C:\Users\jacheung\Dropbox\HLabBackup\Jon\DATA\Behavior\ContLearningCurves';
dataStructLocation = 'C:\Users\jacheung\Dropbox\LocalizationBehavior\DataStructs';
cd(dataStructLocation)
load('BV.mat')
[V] = classifierWrapper_v2(BV,'all','all');

%% Fig 2
%Fig 2E reaction time (first touch to first lick) of population
plotRxnTime(BV)  

%Fig 2F learning curves of population  
learningCurves(behavioralDataLocation)

%Fig 2G psychometric curves of animal during best performing 200 trials
rawPsychometricCurves(BV); 

%Fig 2H discrimation precision of the animal 
discrimination_precision(BV); 

%% Fig 3
%Fig 3A heatmap of whisker motion of 200 best trials in session (sorted)
%can input mouseNumber to plot or variable number
%variable number :1) angle, :3) amplitude :4) midpoint :5) phase
mouseNumber = 9; %paper example mouseNumber 9
variableNumber = 1; %paper example variable 3(amplitude)
plotSortedHeat(BV,mouseNumber,variableNumber);

%Fig 3B plot the amplitude from cue onset
fromCueOnset(BV,3) %3 correlates to the index of amplitude feature

%Fig 3C %outputs whisking changing due to touch for individual and
%population
touchChangesWhisking(BV)

%Fig 3D whisking feature at the peak of each whisk in the cycle. shown in
% Cheung et al. 2019 is angle relative to the discrimination boundary
peakProtractionFeature(BV); 

%Fig 3E Proportion of go/nogo with touches; 
proportionTTypeTouch(BV)

%Fig 3G Scatter of proportion of licking|no touch and licking|touch 
trialProportion(BV,'all');% can set all to 'pro' or 'ret' for touch direction


%% Fig 4
%designMatrix Parameters
params.designvars = 'angle';
% 1) 'theta' 2) 'hilbert' (phase amp midpoint) 3) 'counts' 4) 'ubered'
% 5) 'timing' 6) 'motor' 7) 'decompTime' OR ,'kappa'

params.classes = 'gonogo';
% 1) 'gonogo' 2) 'lick'

% Only for multi-predictor features
params.normalization = 'meanNorm';
% 1) 'meanNorm' 2)'none'

params.dropNonTouch = 'yes';
% 1) 'yes' = drop trials with no touches
% 2) 'no' = keep all trials

feature_distribution(BV,V,params)


%% Fig 5
%Fig A/B Lick probability as a function of touch count 
touchDirection = 'all';
touchOrder = 'all';
numTouchesLickProbability(BV,touchDirection,touchOrder)

%% Fig 6 Radial distance feature isolation task 





%% Fig 7

%Fig 7G prediction heat map 
predictionMatrix = outcomes_heatmap(BV,motorXpreds);

%Fig 7H prediction heat map comaparison
outcomes_heatmap_comparator(predictionMatrix)

