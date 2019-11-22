import tensorflow as tf
from helper_functions import train_on_hdf5_dataset, make_base_model, make_finetuning_model, make_RFD_hdf5, check_hdf5_dataset

# Dataset creation
batch_size = 64
n_epochs = 100
img_shape = (128, 128, 1)
source_imgs_path = './RFD'
n_IDs = 73
emotions = ['angry', 'contemptuous', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
genders = ['male', 'female']
angles = ['090']  # ['000', '045', '090', '135', '180']
dataset_path = './RFD_dataset.h5'
make_RFD_hdf5(emotions=emotions, genders=genders, angles=angles, resized_shape=img_shape[0], im_path=source_imgs_path, output_path=dataset_path)
if 0: check_hdf5_dataset(dataset_path=dataset_path)


# loss & optimizer we will use
classification_loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.optimizers.Adam(1e-4)

# define base CNN model and load if checkpoint exists, otherwise train
CNN_model, ckpt, ckpt_path, manager = make_base_model(img_shape, n_IDs, optimizer, 'CNN_model')
train_on_hdf5_dataset(CNN_model, dataset_path, 'ID_labels', batch_size, n_epochs, classification_loss, optimizer, ckpt, ckpt_path, manager, visualize_batches=False)

# create a new model to finetune and load if checkpoint exists, otherwise train
train_only_decoder = True
finetuning_model_1, finetuning_ckpt_1, finetuning_ckpt_path_1, finetuning_manager_1 = make_finetuning_model(CNN_model, len(emotions), optimizer, train_only_decoder, 'finetuning_model_1')
train_on_hdf5_dataset(finetuning_model_1, dataset_path, 'emotion_labels', batch_size, n_epochs, classification_loss, optimizer, finetuning_ckpt_1, finetuning_ckpt_path_1, finetuning_manager_1)

# create another model to finetune, finetuning all the layers this time. and load if checkpoint exists, otherwise train
train_only_decoder = False
finetuning_model_2, finetuning_ckpt_2, finetuning_ckpt_path_2, finetuning_manager_2 = make_finetuning_model(CNN_model, len(emotions), optimizer, train_only_decoder, 'finetuning_model_2')
train_on_hdf5_dataset(finetuning_model_2, dataset_path, 'emotion_labels', batch_size, n_epochs, classification_loss, optimizer, finetuning_ckpt_2, finetuning_ckpt_path_2, finetuning_manager_2)
