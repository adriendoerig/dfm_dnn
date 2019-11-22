import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from helper_functions import make_base_model, make_finetuning_model, train_step, batch_accuracy
import os


def make_toy_dataset():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
    # also create fake finetuning dataset in which the task is to classify < vs. >= 5.
    finetuning_train_images = train_images.copy()
    finetuning_train_labels = train_labels.copy()
    finetuning_test_images = test_images.copy()
    finetuning_test_labels = test_labels.copy()
    finetuning_train_labels[finetuning_train_labels < 5] = 0
    finetuning_train_labels[finetuning_train_labels >= 5] = 1
    finetuning_test_labels[finetuning_test_labels < 5] = 0
    finetuning_test_labels[finetuning_test_labels >= 5] = 1
    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)
    finetuning_train_labels = tf.keras.utils.to_categorical(finetuning_train_labels, num_classes=10)
    finetuning_test_labels = tf.keras.utils.to_categorical(finetuning_test_labels, num_classes=10)
    # Normalize the images to the range of [0., 1.]
    train_images /= 255.
    test_images /= 255.

    return train_images, train_labels, test_images, test_labels, finetuning_train_images, finetuning_train_labels, finetuning_test_images, finetuning_test_labels


def train_on_toy_dataset(model, train_data, train_labels, test_data, test_labels, batch_size, n_epochs, loss_fn, optimizer, checkpoint, checkpoint_path, saving_manager):

    checkpoint_path = 'toy_' + checkpoint_path
    if os.path.exists(checkpoint_path):
        checkpoint.restore(saving_manager.latest_checkpoint)
    else:
        n_samples = train_data.shape[0]
        n_batches = n_samples//batch_size
        losses = np.zeros(n_batches*n_epochs)
        accuracies = np.zeros(n_batches*n_epochs)
        vars_to_train = model.trainable_variables
        counter = 0

        for epoch in range(n_epochs):
            for batch in range(n_batches):
                batch_imgs = train_data[batch*batch_size:(batch+1)*batch_size]
                batch_labels = train_labels[batch*batch_size:(batch+1)*batch_size]
                losses[counter], accuracies[counter] = train_step(model, batch_imgs, batch_labels, loss_fn, optimizer, vars_to_train)
                if batch % 25 == 0:
                    print('\rEpoch {}, batch {} -- loss = {}, accuracy = {}'.format(epoch, batch, losses[counter], accuracies[counter]), end=' ')
                counter += 1
                checkpoint.step.assign_add(1)
                if int(checkpoint.step) % 250 == 0 or batch == n_batches-1:
                    save_path = saving_manager.save()
                    print("\nSaved checkpoints for step {}: {} (epoch {}).".format(int(checkpoint.step), save_path, epoch))

        if test_data is not None:
            print('\nComputing performance on test set...')
            n_test_samples = test_data.shape[0]
            n_test_batches = n_test_samples//batch_size
            test_loss = 0
            test_accuracy = 0
            for batch in range(n_test_batches):
                batch_imgs = test_data[batch * batch_size:(batch + 1) * batch_size]
                batch_labels = test_labels[batch * batch_size:(batch + 1) * batch_size]
                test_loss += loss_fn(batch_labels, model(batch_imgs))/n_test_batches
                test_accuracy += batch_accuracy(batch_labels, model(batch_imgs))/n_test_batches
                if batch % 25 == 0:
                    print('\rTesting progress: {} %'.format(batch/n_test_batches*100), end=' ')
            print('\nTesting loss = {}\nTesting accuracy = {}'.format(test_loss, test_accuracy))

        fig, a = plt.subplots(1, 2)
        a[0].plot(range(n_batches*n_epochs), losses)
        a[1].plot(range(n_batches*n_epochs), accuracies)
        plt.show()

        return losses, accuracies


# Dataset creation
train_images, train_labels, test_images, test_labels, finetuning_train_images, finetuning_train_labels, finetuning_test_images, finetuning_test_labels = make_toy_dataset()
img_shape = train_images[0].shape
batch_size = 128
n_epochs = 10

# loss & optimizer we will use
classification_loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.optimizers.Adam()

# define base CNN model and load if checkpoint exists, otherwise train
CNN_model, ckpt, ckpt_path, manager = make_base_model(img_shape, 10, optimizer, 'basic_CNN')
train_on_toy_dataset(CNN_model, train_images, train_labels, test_images, test_labels, batch_size, n_epochs, classification_loss, optimizer, ckpt, ckpt_path, manager)

# create a new model to finetune and load if checkpoint exists, otherwise train
train_only_decoder = True
finetuning_model_1, finetuning_ckpt_1, finetuning_ckpt_path_1, finetuning_manager_1 = make_finetuning_model(CNN_model, 2, optimizer, train_only_decoder, 'finetuning_model_1')
train_on_toy_dataset(finetuning_model_1, finetuning_train_images, finetuning_train_labels, finetuning_test_images, finetuning_test_labels,
                 batch_size, n_epochs, classification_loss, optimizer, finetuning_ckpt_1, finetuning_ckpt_path_1, finetuning_manager_1)

# create another model to finetune, finetuning all the layers this time. and load if checkpoint exists, otherwise train
train_only_decoder = False
finetuning_model_2, finetuning_ckpt_2, finetuning_ckpt_path_2, finetuning_manager_2 = make_finetuning_model(CNN_model, 2, optimizer, train_only_decoder, 'finetuning_model_2')
train_on_toy_dataset(finetuning_model_2, finetuning_train_images, finetuning_train_labels, finetuning_test_images, finetuning_test_labels,
                 batch_size, n_epochs, classification_loss, optimizer, finetuning_ckpt_2, finetuning_ckpt_path_2, finetuning_manager_2)
