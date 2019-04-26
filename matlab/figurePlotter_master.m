%%__main__
close all; clear; clc; 
%% Load behavioral data matrix 
behavioralDataLocation = 'C:\Users\jacheung\Dropbox\HLabBackup\Jon\DATA\Behavior\ContLearningCurves';
dataStructLocation = 'C:\Users\jacheung\Dropbox\LocalizationBehavior\DataStructs';
cd(dataStructLocation)
load('BV.mat')
[V] = classifierWrapper_v2(BV,'all','all');

%% FIG 1
%Fig 1F ------------------------------------------------------------------
plotRxnTime(BV)  

%Fig 1G ------------------------------------------------------------------
learningCurves(behavioralDataLocation)

%Fig 1H ------------------------------------------------------------------
rawPsychometricCurves(BV); 

%Fig 1I ------------------------------------------------------------------
discriminationPrecision(BV); 

%% Fig 2
%Fig 2A ------------------------------------------------------------------
%heatmap of whisker motion of 200 best trials in session (sorted)
%can input mouseNumber to plot or variable number
%variable number :1) angle, :3) amplitude :4) midpoint :5) phase

mouseNumber = 9; %paper example mouseNumber 9
variableNumber = 1; %paper example variable 3(amplitude)
plotSortedHeat(BV,mouseNumber,variableNumber);

%Fig 2B ------------------------------------------------------------------
%plot the amplitude from cue onset
fromCueOnset(BV,3)

%Fig 2C ------------------------------------------------------------------
%will plot the whisking feature of the peak of each whisk in the cycle
peakProtractionFeature(BV)

%Fig 1F/2D ------------------------------------------------------------------
%outputs whisking changing due to touch for population and indivdual mice
%along with reaction time (for fig 1F) for all mice
touchChangesWhisking(BV)

%% Fig 3

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

%Fig 3E Proportion of go/nogo with touches; 
proportionTTypeTouch(BV)

%% Fig 4
%Can set to look at protraction touches only. However, a couple mice have
%lots of retraction touches. This will show that mice will lick | no touch.
trialProportion(BV,'all')
%% Fig 5a/b
touchDirection = 'all';
touchOrder = 'all';
numTouchesLickProbability(BV,touchDirection,touchOrder)

%% Fig 6 
