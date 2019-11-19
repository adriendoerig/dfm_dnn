import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os


def make_base_model(img_shape, optimizer, base_model_name):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=img_shape),
        tf.keras.layers.Conv2D(filters=5, kernel_size=(5, 5), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), padding='same', activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')])

    # needed for saving models
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=model)
    ckpt_path = './model_checkpoints/' + base_model_name + '_ckpt'
    manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=1)

    # Print network summary and check which layers are trainable
    model.summary()
    for layer in model.layers:
        print('{}: layer {} is {}.'.format(base_model_name, layer.name, layer.trainable))

    return model, ckpt, ckpt_path, manager


def make_finetuning_model(base_model, n_output_neurons, optimizer, train_only_decoder, finetuning_model_name):
    base_model.layers.pop()  # we remove the last layer from the base model
    base_model.trainable = False if train_only_decoder else True  # choose whether to train the weights of the CNN

    x = base_model.layers[-1].output  # we get the activity in the last layer (the one before the one we just removed)
    x = tf.keras.layers.Dense(n_output_neurons, activation='softmax')(x)  # and add a new fully connected decoder layer
    finetuning_model = tf.keras.models.Model(inputs=base_model.input, outputs=x)

    # setup model saving
    finetuning_ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=finetuning_model)
    finetuning_ckpt_path = './model_checkpoints/' + finetuning_model_name + '_ckpt'
    finetuning_manager = tf.train.CheckpointManager(finetuning_ckpt, finetuning_ckpt_path, max_to_keep=1)

    # Print network summary and check which layers are trainable
    finetuning_model.summary()
    for layer in finetuning_model.layers:
        print('{}: layer {} has trainable =  {}.'.format(finetuning_model_name, layer.name, layer.trainable))

    return finetuning_model, finetuning_ckpt, finetuning_ckpt_path, finetuning_manager

def batch_accuracy(labels, preds):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(preds, axis=1), tf.argmax(labels, axis=1)), tf.float32))


def train_step(model, batch_imgs, batch_labels, loss_fn, optimizer, vars_to_train):
    with tf.GradientTape() as tape:
        preds = model(batch_imgs)
        accuracy = batch_accuracy(batch_labels, preds)
        loss = loss_fn(batch_labels, preds)
        grad = tape.gradient(loss, vars_to_train)
        optimizer.apply_gradients(zip(grad, vars_to_train))
    return loss, accuracy


def train_on_dataset(model, train_data, train_labels, test_data, test_labels, batch_size, n_epochs, loss_fn, optimizer, checkpoint, checkpoint_path, saving_manager):
    if os.path.exists(checkpoint_path):
        checkpoint.restore(checkpoint_path.latest_checkpoint)
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
    train_labels = tf.keras.utils.to_categorical(train_labels)
    test_labels = tf.keras.utils.to_categorical(test_labels)
    finetuning_train_labels = tf.keras.utils.to_categorical(finetuning_train_labels)
    finetuning_test_labels = tf.keras.utils.to_categorical(finetuning_test_labels)
    # Normalize the images to the range of [0., 1.]
    train_images /= 255.
    test_images /= 255.

    return train_images, train_labels, test_images, test_labels, finetuning_train_images, finetuning_train_labels, finetuning_test_images, finetuning_test_labels
