import tensorflow as tf
from helper_functions import train_on_dataset, make_toy_dataset, make_base_model, make_finetuning_model

# Dataset creation
train_images, train_labels, test_images, test_labels, finetuning_train_images, finetuning_train_labels, finetuning_test_images, finetuning_test_labels = make_toy_dataset()
img_shape = train_images[0].shape
batch_size = 128
n_epochs = 10

# loss & optimizer we will use
classification_loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.optimizers.Adam()

# define base CNN model and load if checkpoint exists, otherwise train
CNN_model, ckpt, ckpt_path, manager = make_base_model(img_shape, optimizer, 'basic_CNN')
train_on_dataset(CNN_model, train_images, train_labels, test_images, test_labels, batch_size, n_epochs, classification_loss, optimizer, ckpt, ckpt_path, manager)

# create a new model to finetune and load if checkpoint exists, otherwise train
train_only_decoder = True
finetuning_model_1, finetuning_ckpt_1, finetuning_ckpt_path_1, finetuning_manager_1 = make_finetuning_model(CNN_model, 2, optimizer, train_only_decoder, 'finetuning_model_1')
train_on_dataset(finetuning_model_1, finetuning_train_images, finetuning_train_labels, finetuning_test_images, finetuning_test_labels,
                 batch_size, n_epochs, classification_loss, optimizer, finetuning_ckpt_1, finetuning_ckpt_path_1, finetuning_manager_1)

# create another model to finetune, finetuning all the layers this time. and load if checkpoint exists, otherwise train
train_only_decoder = False
finetuning_model_2, finetuning_ckpt_2, finetuning_ckpt_path_2, finetuning_manager_2 = make_finetuning_model(CNN_model, 2, optimizer, train_only_decoder, 'finetuning_model_2')
train_on_dataset(finetuning_model_2, finetuning_train_images, finetuning_train_labels, finetuning_test_images, finetuning_test_labels,
                 batch_size, n_epochs, classification_loss, optimizer, finetuning_ckpt_2, finetuning_ckpt_path_2, finetuning_manager_2)
