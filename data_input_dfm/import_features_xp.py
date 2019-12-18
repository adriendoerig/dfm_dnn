import os.path as op
from scipy.io import loadmat
import h5py
import numpy as np
from show_trial_experiment import show_trial

# put the task here (either gender, or emotion). Only gender for now
task_string='gender'

# put your path here
base_dir='/home/adf/faghelss/CharestLab/python_dev/humanbased_DNN'

# load .mat containg all data needed
mat_xp_labels = loadmat(op.join(base_dir, f'{task_string}_INPUT_MAT_DNN.mat'))

# (subjects x features x labelstype) tensor containing 1) accuracy and 2) actual responses for presented face
human_classification=mat_xp_labels['human_classification_experiment'] 

# (subjects x trials x features presence/absence) tensor containing presence or absence of features at a given trial
presence_feats_experiment=mat_xp_labels['presence_feats_experiment'] 

# (subjects x trials x stimulus presented) tensor indicating which stimulus was presented 
which_stim_experiment=mat_xp_labels['which_stim_experiment'] 

# a (stimuli x features x pixels) tensor containing the actual images of gabors.
mat_img_gabors = h5py.File(op.join(base_dir,'Gabors_vectorized_images.mat'),'r')

 # un exemple de comment plotter the ith stimulus for participant n along with the response that participant made 
gabors_images=mat_img_gabors.get('gabor_vectorized_images')
gabors_images=np.array(gabors_images)


subject=101 # arbitrary subject to show
trial=2197  # trial to show


correct_incorrect=["incorrect","correct"] # labels for accurate/inaccurate responses 
acc_trial=human_classification[subject,trial,0] # 1st dim for accuracy / 1 == accurate

responses=["male","female"] # label for responses
response_trial=human_classification[subject,trial,1] # 2nd dim for actual responses, also in logical


human_resp=correct_incorrect[acc_trial]  # one can change this for responses if prefered


#this is defined in show_trial_experiment.py ; the script is in the same folder
show_trial(subject, 
               trial,
               human_resp,
               which_stim_experiment,
               presence_feats_experiment,
               gabors_images)

