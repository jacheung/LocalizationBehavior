import scipy.io as sio
import numpy as np
import scipy.signal as sis
import sklearn as skl
from sklearn import linear_model
import matplotlib.pyplot as plt
import copy


def load_data():
    class bdatawrapper:
        def __init__(self, stimulus, response, trialType_matrix):
            """
            Function that takes converted MATLAB structured data from Cheung et al. 2019 and converts it into python class.
            INPUT: (stimulus, response, trialType_matrix)
            OUTPUT: class with nested dictionary for each mouse that contains
                1) var_Names : variables names
                2) ttype_Names : trial outcome for each trial number
                3) S_ctk : stimulus associated with each variable name
                4) R_ntk : spike data
                5) ttype_mat : matrix of trial outcomes
            """
            self.variable_Names = ['0=theta', '1=velocity', '2=amplitude', '3=midpoint', '4=phase', '5=curvature',
                                      '6=M0Adj',
                                      '7=FaxialAdj',
                                      '8=firstTouchIdx', '9=firstTouchOffset', '10=firstTouchAll', '11=lateTouchOnset',
                                      '12=lateTouchOffset',
                                      '13=lateTouchAll', '14=PoleAvailable', '15=lickTimes']
            self.ttype_mat_names = ['hits', 'miss', 'false alarms', 'correct rejection']
            self.S_ctk = stimulus
            self.R_ntk = response
            self.ttype_mat = trialType_matrix

        def simple_touch_feature_distribution(self, variable_number):
            """
            this function will output
            1) predecision_variable_matrix: the features at touch predecision for each mouse
            2) DmatY: go/nogo or lick/nolick matrix x tnum
            """
            ## unroll matrix columnwise. To do it row-wise, don't transpose
            curr_variable = self.S_ctk[variable_number].T.ravel()
            curr_variable_raw = self.S_ctk[variable_number].T

            ## touch indices with all touches after decision masked out
            first_touchIdx = self.S_ctk[8]
            late_touchIdx = self.S_ctk[12]
            first_touch = np.where(first_touchIdx.T.ravel() == 1)
            late_touch = np.where(late_touchIdx.T.ravel() == 1)
            alltIdx = np.hstack((first_touch, late_touch))

            predecision_touch_mat = np.zeros(np.shape(first_touchIdx.T.ravel()))
            predecision_touch_mat[:] = np.nan
            predecision_touch_mat[alltIdx] = 1
            predecision_touch_mat = np.reshape(predecision_touch_mat, (np.shape(first_touchIdx)[1], -1))
            predecision_variable_matrix = (predecision_touch_mat * curr_variable_raw).T

            ## matrix with 2 columns (1: go/nogo 2:lick/nolick) x numTrials
            DmatY = np.vstack((np.sum(self.ttype_mat[[0, 1], :], axis=0),
                               np.sum(self.ttype_mat[[0, 2], :], axis=0))).T

            return predecision_variable_matrix, DmatY

        def predecision_window(self):
            answer_period_window = 450 + 750

            ## finding first lick in each time period
            lix = copy.deepcopy(self.S_ctk[15])
            decision_lick = np.empty((np.shape(self.S_ctk[15])[1], 1))
            for y in range(np.shape(self.S_ctk[15])[1]):
                decision_lick[y] = np.min(
                    np.append(sum(np.where(lix[answer_period_window:, y] == 1), answer_period_window), 4000))
            median_lick_time = np.median(decision_lick[decision_lick < 4000])
            decision_lick[decision_lick == 4000] = median_lick_time
            decision_lick = decision_lick.astype(int)

            ## lick mask for touch indices
            lick_mask = np.zeros(np.shape(lix))
            lick_mask[:] = np.nan
            for y in range(np.shape(lick_mask)[1]):
                lick_mask[:decision_lick[y][0], y] = 1

            return lick_mask

        def whisking_peaks_and_troughs(self):
            theta = copy.deepcopy(self.S_ctk[0])
            amplitude = self.S_ctk[2]>5
            theta[~amplitude]=np.nan

            peak_times_matrix= np.empty(np.shape(theta))
            peak_times_matrix[:] = np.nan
            trough_times_matrix = copy.deepcopy(peak_times_matrix)

            for g in range(np.shape(theta)[1]):
                curr_trial = theta[:, g]
                peak_times = sis.find_peaks(curr_trial, distance=15, width=3)[0]
                trough_times = sis.find_peaks(curr_trial*-1, distance=15, width=3)[0]
                peak_times_matrix[peak_times, g] = 1
                trough_times_matrix[trough_times, g] = 1

            return peak_times_matrix, trough_times_matrix

    U={}

    for y in range(1, 11):
        dirAd = f"/Users/jonathancheung/Documents/Github/LocalizationBehavior/python/dataStructs/ms{y}.mat"
        b_dat = sio.loadmat(dirAd)
        currMsDat = b_dat['ms']
        S_ctk = currMsDat[0, 0:16]
        R_ntk = currMsDat[0, 16]
        ttype_mat_tmp = currMsDat[0, 17]
        ttype_mat = np.reshape(ttype_mat_tmp, (4, currMsDat[0, 1].shape[1]))
        U[f'ms{y}'] = bdatawrapper(S_ctk, R_ntk, ttype_mat)

    return U