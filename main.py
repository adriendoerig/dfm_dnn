############################### NOTES ###############################
# OVERFITTING POSSIBILITY IF DIFFERENT GAZES IN TRAIN & TEST SETS
#####################################################################

import tensorflow as tf
import numpy as np
from helper_functions import train_on_hdf5_dataset, make_base_model, make_finetuning_model, make_RFD_hdf5, \
    make_humanbased_hdf5, check_hdf5_dataset, test_robustness, bar_plot


############################### NOTES ###############################
# PARAMETERS, DATASETS, ETC.
#####################################################################


# General parameters
model_name = 'basic_CNN'  # 'resnet50' or 'vgg' for standard networks. Otherwise, builds a simple CNN (see make_base_model() in helper_functions.py)
model_counter = 0  # will be used to label models in increasing order
batch_size = 64
n_epochs = 100
img_shape = (128, 128, 1)
RFD_IDs_model_names = []  # we will store the names of the models we test on the RFD identities here
RFD_IDs_test_accuracies = []  # we will store the accuracies of the different models here

# Radboud faces dataset
source_imgs_path = './RFD'
n_IDs = 73
emotions = ['angry', 'contemptuous', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
genders = ['male', 'female']
angles = ['090']  # ['000', '045', '090', '135', '180']
RFD_dataset_path = './RFD_dataset.h5'
make_RFD_hdf5(emotions=emotions, genders=genders, angles=angles, resized_shape=img_shape[0], im_path=source_imgs_path, output_path=RFD_dataset_path)
if 0: check_hdf5_dataset('RFD', dataset_path=RFD_dataset_path)

# Humanbased dataset
task = 'gender'
subject = 1
make_humanbased_hdf5(task, subject, resized_shape=img_shape[0])
humanbased_dataset_path = './humanbased_dataset_{}_subject_{}.h5'.format(task, subject) if subject is not None else './humanbased_dataset_{}_all_subjects.h5'.format(task)
if 0: check_hdf5_dataset('humanbased', dataset_path=humanbased_dataset_path)

# loss & optimizer we will use
classification_loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.optimizers.Adam(1e-4)


############################### NOTES ######################################
# PHASE 0: TRAINING BASE MODELS ON RFD IDENTITIES, AND ON HUMANBASED DATA
############################################################################


# define base CNN model and load if checkpoint exists, otherwise train on RFD
print('\n\nBASE MODEL TRAINED ON RFD IDENTITIES')
base_model, ckpt_dict_base_model = make_base_model(img_shape, n_IDs, optimizer, model_name, model_counter)
test_acc = train_on_hdf5_dataset(base_model, RFD_dataset_path, 'ID_labels', batch_size, n_epochs, classification_loss, optimizer, ckpt_dict_base_model, visualize_batches=False)
RFD_IDs_test_accuracies.append(test_acc)
RFD_IDs_model_names.append(str(model_counter)+'_base_model')
model_counter += 1

# same, but on humanbased data
print('\n\nBASE MODEL TRAINED ON HUMAN BASED DATA')
base_model_HB, ckpt_dict_base_model_HB = make_base_model(img_shape, 2, optimizer, model_name+'_HB', model_counter)
train_on_hdf5_dataset(base_model_HB, humanbased_dataset_path, 'humanbased_labels', batch_size, n_epochs, classification_loss, optimizer, ckpt_dict_base_model_HB, visualize_batches=False)
model_counter += 1


############################### NOTES ###############################
# PHASE 1: FINETUNING ON RADBOUD FACES DATASET GENDERS
#####################################################################


# create a new model to finetune ON GENDER and load if checkpoint exists, otherwise train on RFD
train_only_decoder = True
print('\n\nBASE MODEL = RFD_IDENTITIES, FINETUNING ON RFD_GENDERS -- ONLY LAST LAYER IS FINETUNED')
b_RFDid_fl_RFDg, ckpt_dict_b_RFDid_fl_RFDg = make_finetuning_model(base_model, len(genders), optimizer, train_only_decoder, 'b_RFDid_fl_RFDg', model_counter)  # name stands for basemodel_RFDidentities_finetunelast_RFDgenders
train_on_hdf5_dataset(b_RFDid_fl_RFDg, RFD_dataset_path, 'gender_labels', batch_size, n_epochs, classification_loss, optimizer, ckpt_dict_b_RFDid_fl_RFDg)
model_counter += 1

print('\n\nBASE MODEL = RFD_IDENTITIES, FINETUNING ON RFD_GENDERS -- ALL LAYERS ARE FINETUNED')
# create another model to finetune ON GENDER, finetuning all the layers this time. and load if checkpoint exists, otherwise train
train_only_decoder = False
b_RFDid_fa_RFDg, ckpt_dict_b_RFDid_fa_RFDg = make_finetuning_model(base_model, len(genders), optimizer, train_only_decoder, 'b_RFDid_fa_RFDg', model_counter)  # name stands for basemodel_RFDidentities_finetuneall_RFDgenders
train_on_hdf5_dataset(b_RFDid_fa_RFDg, RFD_dataset_path, 'gender_labels', batch_size, n_epochs, classification_loss, optimizer, ckpt_dict_b_RFDid_fa_RFDg)
model_counter += 1


############################### NOTES ###############################
# PHASE 2: FINETUNING ON THE EXPERIMENTAL HUMAN RESPONSES
#####################################################################


# create a new model to finetune ON HUMANBASED LABELS and load if checkpoint exists, otherwise train
print('\n\nBASE MODEL = RFD_IDENTITIES, FINETUNING ON HUMANBASED_GENDERS -- ONLY LAST LAYER IS FINETUNED')
train_only_decoder = True
b_RFDid_fl_HBg, ckpt_dict_b_RFDid_fl_HBg = make_finetuning_model(base_model, 2, optimizer, train_only_decoder, 'b_RFDid_fl_HBg', model_counter)    # name stands for basemodel_RFDidentities_finetunelast_HumanBasedgenders
train_on_hdf5_dataset(b_RFDid_fl_HBg, humanbased_dataset_path, 'humanbased_labels', batch_size, n_epochs, classification_loss, optimizer, ckpt_dict_b_RFDid_fl_HBg)
model_counter += 1

# create another model to finetune ON HUMANBASED LABELS, finetuning all the layers this time. and load if checkpoint exists, otherwise train
print('\n\nBASE MODEL = RFD_IDENTITIES, FINETUNING ON HUMANBASED_GENDERS -- ALL LAYERS ARE IS FINETUNED')
train_only_decoder = False
b_RFDid_fa_HBg, ckpt_dict_b_RFDid_fa_HBg = make_finetuning_model(base_model, 2, optimizer, train_only_decoder, 'b_RFDid_fa_HBg', model_counter)
train_on_hdf5_dataset(b_RFDid_fa_HBg, humanbased_dataset_path, 'humanbased_labels', batch_size, n_epochs, classification_loss, optimizer, ckpt_dict_b_RFDid_fa_HBg)
model_counter += 1


############################### NOTES ######################################################
# PHASE 3: RETRAIN THE LAST LAYER OF EACH MODEL TO PERFORM THE IDENTITY TASK ON RFD
############################################################################################


train_only_decoder = True

print('\n\nBASE MODEL = HUMANBASED_GENDERS -- FINETUNING LAST LAYER BACK TO RFD_IDENTITIES')
base_model_HB_IDs, ckpt_dict_base_model_humanbased_IDs = make_finetuning_model(base_model_HB, n_IDs, optimizer, train_only_decoder, model_name+'_HB_IDs', model_counter)
test_acc = train_on_hdf5_dataset(base_model_HB_IDs, RFD_dataset_path, 'ID_labels', batch_size, n_epochs, classification_loss, optimizer, ckpt_dict_base_model_humanbased_IDs)
RFD_IDs_test_accuracies.append(test_acc)
RFD_IDs_model_names.append(str(model_counter)+'_base_model_HB_IDs')
model_counter += 1

print('\n\nBASE MODEL = RFD_IDENTITIES LAST LAYER FINETUNED ON RFD_GENDERS -- FINETUNING LAST LAYER BACK TO RFD_IDENTITIES')
b_RFDid_fl_RFDg_IDs, ckpt_dict_b_RFDid_fl_RFDg_IDs = make_finetuning_model(b_RFDid_fl_RFDg, n_IDs, optimizer, train_only_decoder, 'b_RFDid_fl_RFDg_IDs', model_counter)
test_acc = train_on_hdf5_dataset(b_RFDid_fl_RFDg_IDs, RFD_dataset_path, 'ID_labels', batch_size, n_epochs, classification_loss, optimizer, ckpt_dict_b_RFDid_fl_RFDg_IDs)
RFD_IDs_test_accuracies.append(test_acc)
RFD_IDs_model_names.append(str(model_counter)+'_b_RFDid_fl_RFDg_IDs')
model_counter += 1

print('\n\nBASE MODEL = RFD_IDENTITIES ALL LAYERS FINETUNED ON RFD_GENDERS -- FINETUNING LAST LAYER BACK TO RFD_IDENTITIES')
b_RFDid_fa_RFDg_IDs, ckpt_dict_b_RFDid_fa_RFDg_IDs = make_finetuning_model(b_RFDid_fa_RFDg, n_IDs, optimizer, train_only_decoder, 'b_RFDid_fa_RFDg_IDs', model_counter)
test_acc = train_on_hdf5_dataset(b_RFDid_fa_RFDg_IDs, RFD_dataset_path, 'ID_labels', batch_size, n_epochs, classification_loss, optimizer, ckpt_dict_b_RFDid_fa_RFDg_IDs)
RFD_IDs_test_accuracies.append(test_acc)
RFD_IDs_model_names.append(str(model_counter)+'_b_RFDid_fa_RFDg_IDs')
model_counter += 1

print('\n\nBASE MODEL = RFD_IDENTITIES LAST LAYER FINETUNED ON HUMANBASED_GENDERS -- FINETUNING LAST LAYER BACK TO RFD_IDENTITIES')
b_RFDid_fl_HBg_IDs, ckpt_dict_b_RFDid_fl_HBg_IDs = make_finetuning_model(b_RFDid_fl_HBg, n_IDs, optimizer, train_only_decoder, 'b_RFDid_fl_HBg_IDs', model_counter)
test_acc = train_on_hdf5_dataset(b_RFDid_fl_HBg_IDs, RFD_dataset_path, 'ID_labels', batch_size, n_epochs, classification_loss, optimizer, ckpt_dict_b_RFDid_fl_HBg_IDs)
RFD_IDs_test_accuracies.append(test_acc)
RFD_IDs_model_names.append(str(model_counter)+'_b_RFDid_fl_HBg_IDs')
model_counter += 1

print('\n\nBASE MODEL = RFD_IDENTITIES ALL LAYERS FINETUNED ON HUMANBASED_GENDERS -- FINETUNING LAST LAYER BACK TO RFD_IDENTITIES')
b_RFDid_fa_HBg_IDs, ckpt_dict_b_RFDid_fa_HBg_IDs = make_finetuning_model(b_RFDid_fa_HBg, n_IDs, optimizer, train_only_decoder, 'b_RFDid_fa_HBg_IDs', model_counter)
test_acc = train_on_hdf5_dataset(b_RFDid_fa_HBg_IDs, RFD_dataset_path, 'ID_labels', batch_size, n_epochs, classification_loss, optimizer, ckpt_dict_b_RFDid_fa_HBg_IDs)
RFD_IDs_test_accuracies.append(test_acc)
RFD_IDs_model_names.append(str(model_counter)+'_b_RFDid_fa_HBg_IDs')
model_counter += 1

print('\n\nCONTROL: BASE MODEL = RANDOMLY CONNECTED NETWORK -- FINETUNING LAST LAYER TO RFD_IDENTITIES')
control_model, ckpt_dict_control_model = make_base_model(img_shape, n_IDs, optimizer, model_name, model_counter)
control_model_IDs, ckpt_dict_control_model_IDs = make_finetuning_model(control_model, n_IDs, optimizer, train_only_decoder, 'control_model_IDs', model_counter)
test_acc = train_on_hdf5_dataset(control_model_IDs, RFD_dataset_path, 'ID_labels', batch_size, n_epochs, classification_loss, optimizer, ckpt_dict_control_model_IDs)
RFD_IDs_test_accuracies.append(test_acc)
RFD_IDs_model_names.append(str(model_counter)+'_control_model')
model_counter += 1

bar_plot(np.expand_dims(RFD_IDs_test_accuracies, -1), RFD_IDs_model_names, ['test_dataset'], './model_checkpoints/_test_accuracies.png', x_label='Models', y_label='Test accuracy', title='all model accuracies on test set')

############################### NOTES #############################################################
# PHASE 4: TEST ROBUSTNESS OF THE DIFFERENT MODELS TO ADVERSARIAL NOISE ON THE IDENTITY TASK ON RFD
###################################################################################################


models_list = [base_model, base_model_HB_IDs, b_RFDid_fl_RFDg_IDs, b_RFDid_fa_RFDg_IDs, b_RFDid_fl_HBg_IDs, b_RFDid_fa_HBg_IDs, control_model_IDs]
model_checkpoint_dicts_list = [ckpt_dict_base_model, ckpt_dict_base_model_HB, ckpt_dict_b_RFDid_fl_RFDg_IDs, ckpt_dict_b_RFDid_fa_RFDg_IDs, ckpt_dict_b_RFDid_fl_HBg_IDs, ckpt_dict_b_RFDid_fa_HBg_IDs, ckpt_dict_control_model_IDs]
adversarial_accuracies = np.zeros((len(models_list), 3))  # n_models x [clean_acc, fgm_attack_acc, pgd_attack_acc]

for n in range(len(RFD_IDs_model_names)):
    adversarial_accuracies[n, 0], adversarial_accuracies[n, 1], adversarial_accuracies[n, 2] = test_robustness(models_list[n], RFD_IDs_model_names[n], model_checkpoint_dicts_list[n], RFD_dataset_path, eps=0.3)

bar_plot(adversarial_accuracies, RFD_IDs_model_names, ['clean', 'fgm_attack', 'pgd_attack'], './model_checkpoints/_adversarial_test_accuracies.png', x_label='Models', y_label='Test accuracy', title='all model accuracies on test set')
