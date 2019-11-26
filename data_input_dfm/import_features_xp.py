from scipy.io import loadmat
import os.path as op
import h5py as h5py

# put the task here (either gender, or emotion). Only gender for now
task_string='gender'

# put your path here
base_dir='/home/adf/faghelss/CharestLab/python_dev/humanbased_DNN'

# load .mat containg all data needed
mat_xp_labels = loadmat(op.join(base_dir, f'{task_string}_INPUT_MAT_DNN.mat'))

# (subjects x features x labelstype) tensor containing 1) accuracy and 2) actual responses for presented face
human_classification=mat_xp_labels['human_classification_experiment'] 

# (subjects x trials x presence/absence) tensor containing presence or absence of features at a given trial
presence_feats_experiment=mat_xp_labels['presence_feats_experiment'] 

# (subjects x trials x stimulus presented) tensor indicating which stimulus was presented 
which_stim_experiment=mat_xp_labels['which_stim_experiment'] 

# a (stimuli x features x pixels) tensor containing the actual images of gabors.
mat_img_gabors = h5py.File(op.join(base_dir,'Gabors_vectorized_images.mat'),'r')

gabors_images=mat_img_gabors['gabor_vectorized_images']