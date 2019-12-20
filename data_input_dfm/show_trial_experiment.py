import numpy as np
import matplotlib.pyplot as plt

def show_trial(subject, trial, human_resp, which_stim_experiment, presence_feats_experiment, gabors_images):
    that_stim = int(which_stim_experiment[subject, trial]) - 1  # gets stimulus, -1 changes matlab indexes to python indexes
    # choose the specific trial, transform in boolean
    these_gabors = presence_feats_experiment[subject, trial, :] == 1
    # get all the Gabors images that we presented at this trial, for that stimulus
    stim_ex = gabors_images[that_stim, these_gabors, :]
    stim = np.sum(stim_ex, 0)  # sum all Gabs images along first dimension
    stim = np.reshape(stim, [250, 250])  # reshape the vectorized image to actual 250x250 image
    stim = np.array(stim).T  # transpose required from matlab to python indexing
    print(stim.shape)
    plt.imshow(stim, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title(human_resp)
    plt.show()
