############################### NOTES ###############################
# OVERFITTING POSSIBILITY IF DIFFERENT GAZES IN TRAIN & TEST SETS
#####################################################################

import tensorflow as tf
from helper_functions import train_on_hdf5_dataset, make_base_model, make_finetuning_model, make_RFD_hdf5, make_humanbased_hdf5, check_hdf5_dataset


############################### NOTES ###############################
# PARAMETERS, DATASETS, ETC.
#####################################################################


# General parameters
model_name = 'basic_CNN'  # 'resnet50' or 'vgg' for standard networks. Otherwise, builds a simple CNN (see make_base_model() in helper_functions.py)
batch_size = 64
n_epochs = 100
img_shape = (128, 128, 1)

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
CNN_model, CNN_ckpt_dict = make_base_model(img_shape, n_IDs, optimizer, model_name)
train_on_hdf5_dataset(CNN_model, RFD_dataset_path, 'ID_labels', batch_size, n_epochs, classification_loss, optimizer, CNN_ckpt_dict, visualize_batches=False)

# same, but on humanbased data
print('\n\nBASE MODEL TRAINED ON HUMAN BASED DATA')
CNN_model_HB, ckpt_dict_HB = make_base_model(img_shape, 2, optimizer, model_name+'_HB')
train_on_hdf5_dataset(CNN_model_HB, humanbased_dataset_path, 'humanbased_labels', batch_size, n_epochs, classification_loss, optimizer, ckpt_dict_HB, visualize_batches=False)


############################### NOTES ###############################
# PHASE 1: FINETUNING ON RADBOUD FACES DATASET GENDERS
#####################################################################


# create a new model to finetune ON GENDER and load if checkpoint exists, otherwise train on RFD
train_only_decoder = True
print('\n\nBASE MODEL = RFD_IDENTITIES, FINETUNING ON RFD_GENDERS -- ONLY LAST LAYER IS FINETUNED')
finetuning_model_1, ckpt_dict_fintune_1 = make_finetuning_model(CNN_model, len(genders), optimizer, train_only_decoder, 'finetuning_model_1')
train_on_hdf5_dataset(finetuning_model_1, RFD_dataset_path, 'gender_labels', batch_size, n_epochs, classification_loss, optimizer, ckpt_dict_fintune_1)

print('\n\nBASE MODEL = RFD_IDENTITIES, FINETUNING ON RFD_GENDERS -- ALL LAYERS ARE FINETUNED')
# create another model to finetune ON GENDER, finetuning all the layers this time. and load if checkpoint exists, otherwise train
train_only_decoder = False
finetuning_model_2, ckpt_dict_fintune_2 = make_finetuning_model(CNN_model, len(genders), optimizer, train_only_decoder, 'finetuning_model_2')
train_on_hdf5_dataset(finetuning_model_2, RFD_dataset_path, 'gender_labels', batch_size, n_epochs, classification_loss, optimizer, ckpt_dict_fintune_2)


############################### NOTES ###############################
# PHASE 2: FINETUNING ON THE EXPERIMENTAL HUMAN RESPONSES
#####################################################################


# create a new model to finetune ON HUMANBASED LABELS and load if checkpoint exists, otherwise train
print('\n\nBASE MODEL = RFD_IDENTITIES, FINETUNING ON HUMANBASED_GENDERS -- ONLY LAST LAYER IS FINETUNED')
train_only_decoder = True
finetuning_model_HB1, ckpt_dict_fintune_HB1 = make_finetuning_model(CNN_model, 2, optimizer, train_only_decoder, 'finetuning_model_HB1')
train_on_hdf5_dataset(finetuning_model_HB1, humanbased_dataset_path, 'humanbased_labels', batch_size, n_epochs, classification_loss, optimizer, ckpt_dict_fintune_HB1)

# create another model to finetune ON HUMANBASED LABELS, finetuning all the layers this time. and load if checkpoint exists, otherwise train
print('\n\nBASE MODEL = RFD_IDENTITIES, FINETUNING ON HUMANBASED_GENDERS -- ALL LAYERS ARE IS FINETUNED')
train_only_decoder = False
finetuning_model_HB2, ckpt_dict_fintune_HB2 = make_finetuning_model(CNN_model, 2, optimizer, train_only_decoder, 'finetuning_model_HB2')
train_on_hdf5_dataset(finetuning_model_HB2, humanbased_dataset_path, 'humanbased_labels', batch_size, n_epochs, classification_loss, optimizer, ckpt_dict_fintune_HB2)


############################### NOTES ######################################################
# PHASE 3: RETRAIN THE LAST LAYER OF EACH MODEL TO PERFORM THE IDENTITY TASK ON RFD
############################################################################################


train_only_decoder = True

print('\n\nBASE MODEL = RFD_IDENTITIES LAST LAYER FINETUNED ON RFD_GENDERS -- FINETUNING LAST LAYER BACK TO RFD_IDENTITIES')
finetuning_model_1_IDs, ckpt_dict_fintune_1_IDs = make_finetuning_model(finetuning_model_1, n_IDs, optimizer, train_only_decoder, 'finetuning_model_1_IDs')
train_on_hdf5_dataset(finetuning_model_1_IDs, RFD_dataset_path, 'ID_labels', batch_size, n_epochs, classification_loss, optimizer, ckpt_dict_fintune_1_IDs)

print('\n\nBASE MODEL = RFD_IDENTITIES ALL LAYERS FINETUNED ON RFD_GENDERS -- FINETUNING LAST LAYER BACK TO RFD_IDENTITIES')
finetuning_model_2_IDs, ckpt_dict_fintune_2_IDs = make_finetuning_model(finetuning_model_2, n_IDs, optimizer, train_only_decoder, 'finetuning_model_2_IDs')
train_on_hdf5_dataset(finetuning_model_2_IDs, RFD_dataset_path, 'ID_labels', batch_size, n_epochs, classification_loss, optimizer, ckpt_dict_fintune_2_IDs)

print('\n\nBASE MODEL = RFD_IDENTITIES LAST LAYER FINETUNED ON HUMANBASED_GENDERS -- FINETUNING LAST LAYER BACK TO RFD_IDENTITIES')
finetuning_model_HB1_IDs, ckpt_dict_fintune_HB1_IDs = make_finetuning_model(finetuning_model_HB1, n_IDs, optimizer, train_only_decoder, 'finetuning_model_HB1_IDs')
train_on_hdf5_dataset(finetuning_model_HB1_IDs, RFD_dataset_path, 'ID_labels', batch_size, n_epochs, classification_loss, optimizer, ckpt_dict_fintune_HB1_IDs)

print('\n\nBASE MODEL = RFD_IDENTITIES ALL LAYERS FINETUNED ON HUMANBASED_GENDERS -- FINETUNING LAST LAYER BACK TO RFD_IDENTITIES')
finetuning_model_HB2_IDs, ckpt_dict_fintune_HB2_IDs = make_finetuning_model(finetuning_model_HB2s, n_IDs, optimizer, train_only_decoder, 'finetuning_model_HB2_IDs')
train_on_hdf5_dataset(finetuning_model_HB2_IDs, RFD_dataset_path, 'ID_labels', batch_size, n_epochs, classification_loss, optimizer, ckpt_dict_fintune_HB2_IDs)



############################### NOTES #############################################################
# PHASE 4: TEST ROBUSTNESS OF THE DIFFERENT MODELS TO ADVERSARIAL NOISE ON THE IDENTITY TASK ON RFD
###################################################################################################

models_list = [CNN_model, finetuning_model_1_IDs, finetuning_model_2_IDs, finetuning_model_HB1_IDs, finetuning_model_HB2_IDs]
model_checkpoint_dicts_list = [CNN_ckpt_dict, ckpt_dict_fintune_1_IDs, ckpt_dict_fintune_2_IDs, ckpt_dict_fintune_HB1_IDs, ckpt_dict_fintune_HB2_IDs]
model_names_list = ['base_model', 'finetuning_model_1_IDs', 'finetuning_model_2_IDs', 'finetuning_model_HB1_IDs', 'finetuning_model_HB2_IDs']

for n in range(len(model_names_list)):
    test_robustness(models_list[n], model_names_list[n], model_checkpoint_dicts_list[n], dataset_path, eps=0.3)
