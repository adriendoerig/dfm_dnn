############################### NOTES ###############################
# OVERFITTING POSSIBILITY IF DIFFERENT GAZES IN TRAIN & TEST SETS
#####################################################################

import tensorflow as tf
from helper_functions import train_on_hdf5_dataset, make_base_model, make_finetuning_model, make_RFD_hdf5, make_humanbased_hdf5, check_hdf5_dataset


############################### NOTES ###############################
# PARAMETERS, DATASETS, ETC.
#####################################################################


# General parameters
batch_size = 64
n_epochs = 100
img_shape = (128, 128, 1)

# Radboud faces dataset
source_imgs_path = './RFD'
n_IDs = 73
emotions = ['angry', 'contemptuous', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
genders = ['male', 'female']
angles = ['090']  # ['000', '045', '090', '135', '180']
dataset_path = './RFD_dataset.h5'
make_RFD_hdf5(emotions=emotions, genders=genders, angles=angles, resized_shape=img_shape[0], im_path=source_imgs_path, output_path=dataset_path)
if 0: check_hdf5_dataset('RFD', dataset_path=dataset_path)

# Humanbased dataset
task = 'gender'
subject = 1
make_humanbased_hdf5(task, subject)
if 0: check_hdf5_dataset('humanbased', dataset_path='./humanbased_dataset_{}_subject_{}.h5'.format(task, subject))

# loss & optimizer we will use
classification_loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.optimizers.Adam(1e-4)


############################### NOTES ###############################
# PHASE 0: TRAINING A BASE MODEL ON RADBOUD FACES DATASET IDENTITIES
#####################################################################


# define base CNN model and load if checkpoint exists, otherwise train on RFD
CNN_model, ckpt, ckpt_path, manager = make_base_model(img_shape, n_IDs, optimizer, 'CNN_model')
train_on_hdf5_dataset(CNN_model, dataset_path, 'ID_labels', batch_size, n_epochs, classification_loss, optimizer, ckpt, ckpt_path, manager, visualize_batches=False)


############################### NOTES ###############################
# PHASE 1: FINETUNING ON RADBOUD FACES DATASET GENDERS
#####################################################################


# create a new model to finetune ON GENDER and load if checkpoint exists, otherwise train on RFD
train_only_decoder = True
finetuning_model_1, finetuning_ckpt_1, finetuning_ckpt_path_1, finetuning_manager_1 = make_finetuning_model(CNN_model, len(genders), optimizer, train_only_decoder, 'finetuning_model_1')
train_on_hdf5_dataset(finetuning_model_1, dataset_path, 'gender_labels', batch_size, n_epochs, classification_loss, optimizer, finetuning_ckpt_1, finetuning_ckpt_path_1, finetuning_manager_1)

# create another model to finetune ON GENDER, finetuning all the layers this time. and load if checkpoint exists, otherwise train
train_only_decoder = False
finetuning_model_2, finetuning_ckpt_2, finetuning_ckpt_path_2, finetuning_manager_2 = make_finetuning_model(CNN_model, len(genders), optimizer, train_only_decoder, 'finetuning_model_2')
train_on_hdf5_dataset(finetuning_model_2, dataset_path, 'gender_labels', batch_size, n_epochs, classification_loss, optimizer, finetuning_ckpt_2, finetuning_ckpt_path_2, finetuning_manager_2)


############################### NOTES ###############################
# PHASE 2: FINETUNING ON THE EXPERIMENTAL HUMAN RESPONSES
#####################################################################


# create a new model to finetune ON HUMANBASED LABELS and load if checkpoint exists, otherwise train
train_only_decoder = True
finetuning_model_HB1, finetuning_ckpt_HB1, finetuning_ckpt_path_HB1, finetuning_manager_HB1 = make_finetuning_model(CNN_model, 2, optimizer, train_only_decoder, 'finetuning_model_HB1')
train_on_hdf5_dataset(finetuning_model_HB1, dataset_path, 'humanbased_labels', batch_size, n_epochs, classification_loss, optimizer, finetuning_ckpt_HB1, finetuning_ckpt_path_HB1, finetuning_manager_HB1)

# create another model to finetune ON HUMANBASED LABELS, finetuning all the layers this time. and load if checkpoint exists, otherwise train
train_only_decoder = False
finetuning_model_HB2, finetuning_ckpt_HB2, finetuning_ckpt_path_HB2, finetuning_manager_HB2 = make_finetuning_model(CNN_model, 2, optimizer, train_only_decoder, 'finetuning_model_HB2')
train_on_hdf5_dataset(finetuning_model_HB2, dataset_path, 'humanbased_labels', batch_size, n_epochs, classification_loss, optimizer, finetuning_ckpt_2, finetuning_ckpt_path_2, finetuning_manager_2)
