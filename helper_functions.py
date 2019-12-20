import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from skimage.transform import resize
import imgaug.augmenters as iaa
import os, random, h5py
import os.path as op
from scipy.io import loadmat


def make_base_model(img_shape, n_output_neurons, optimizer, base_model_name):

    if base_model_name.lower() == 'resnet50':
        from tensorflow.keras.applications import ResNet50
        from main import batch_size
        input_layer = tf.keras.layers.Input(batch_shape=(batch_size,)+img_shape)
        model = ResNet50(weights=None, input_tensor=input_layer, classes=n_output_neurons)
    elif base_model_name.lower() == 'vgg16' and img_shape[0]:
        try:
            from tensorflow.keras.applications import VGG16
            from main import batch_size
            input_layer = tf.keras.layers.Input(batch_shape=(batch_size,)+img_shape)
            model = VGG16(weights=None, input_tensor=input_layer, classes=n_output_neurons)
        except:
            print('VGGnet requires with 224x224x3 images. Please choose another network')
    else:
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=img_shape),
            tf.keras.layers.Conv2D(filters=25, kernel_size=(9, 9), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(filters=50, kernel_size=(5, 5), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(filters=50, kernel_size=(5, 5), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(filters=100, kernel_size=(3, 3), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(n_output_neurons, activation='softmax')])

    # needed for saving models
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=model)
    ckpt_path = './model_checkpoints/' + base_model_name + '_ckpt'
    manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=1)

    # Print network summary and check which layers are trainable
    model.summary()
    for layer in model.layers:
        print('{}: layer {} has trainable = {}.'.format(base_model_name, layer.name, layer.trainable))

    return model, ckpt, ckpt_path, manager


def make_finetuning_model(base_model, n_output_neurons, optimizer, train_only_decoder, finetuning_model_name):

    cloned_model = tf.keras.models.clone_model(base_model)  # we need to make a new copy of the model to avoid changing the weights of the base models when we train the finetuning model.
    cloned_model.trainable = False if train_only_decoder else True  # choose whether to train the weights of the CNN

    x = cloned_model.layers[-2].output  # we get the activity in the last layer before the decoder.
    x = tf.keras.layers.Dense(n_output_neurons, activation='softmax')(x)  # and add a new fully connected decoder layer on top of that.
    finetuning_model = tf.keras.models.Model(inputs=cloned_model.input, outputs=x)

    # setup model saving
    finetuning_ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=finetuning_model)
    finetuning_ckpt_path = './model_checkpoints/' + finetuning_model_name + '_ckpt'
    finetuning_manager = tf.train.CheckpointManager(finetuning_ckpt, finetuning_ckpt_path, max_to_keep=1)

    # Print network summary and check which layers are trainable
    finetuning_model.summary()
    for layer in finetuning_model.layers:
        print('{}: layer {} has trainable = {}.'.format(finetuning_model_name, layer.name, layer.trainable))

    return finetuning_model, finetuning_ckpt, finetuning_ckpt_path, finetuning_manager


def make_RFD_hdf5(emotions='all', genders='all', angles='all', resized_shape=224, im_path='./RFD', output_path='./RFD_dataset.h5'):
    '''load data from the Radboud Faces Dataset.
       emotions: a list of 'angry', 'contemptuous', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised' or 'all'
       genders: a list of 'male', 'female' or 'all'
       angles: a list of '000', '045', '090', '135', '180' or 'all'
       path: path to the dataset (e.g. './RFD')
       resized_shape: int we will resize images to be squares of this size
       '''

    if os.path.exists(output_path):
        print('file {} exists. Skipping dataset creation'.format(output_path))
        return
    # make sure we have lists (to cope with 'angry' instead of ['angry'] for example)
    emotions = emotions if isinstance(emotions, list) else [emotions]
    genders = genders if isinstance(genders, list) else [genders]
    angles = angles if isinstance(angles, list) else [angles]
    # 'all' means you want all the possibilities
    emotions = ['angry', 'contemptuous', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised'] if emotions == ['all'] else emotions
    genders = ['male', 'female'] if genders == ['all'] else genders
    angles = ['000', '045', '090', '135', '180'] if angles == ['all'] else angles
    # 'male' and 'female' are both part of 'female', so we use '_male' instead
    genders = ['_male' if gender == 'male' else gender for gender in genders]
    # get list of images satisfying the requested features
    full_list = os.listdir(im_path)
    final_list = [nm for nm in full_list if (any(ft in nm for ft in emotions) and any(ft in nm for ft in genders) and any(ft in nm for ft in angles))]  # this complicated statement makes sure at least one requested emotion, one requested gender and one requested angle are present
    random.shuffle(final_list)
    n_imgs, n_IDs, n_emotions, n_genders = len(final_list), 73, len(emotions), len(genders)

    set_fractions = {'train': [0.0, 0.6], 'val': [0.6, 0.8], 'test': [0.8, 1.0]}  # fraction of all data for each set

    # go through all images, crop to a square image in the face region and resize to manageable size, get labels and save
    with h5py.File(output_path, "w") as f:
        for set in ["train", "val", "test"]:
            img_indices = [int(set_fractions[set][0]*n_imgs), int(set_fractions[set][1]*n_imgs)]
            set_n_imgs = img_indices[1] - img_indices[0]
            group = f.create_group(set)
            group.create_dataset("n_imgs", data=[set_n_imgs])
            group.create_dataset("data", shape=((np.ceil(set_fractions[set][1]-set_fractions[set][0])*n_imgs), resized_shape, resized_shape, 1), dtype=np.float32)
            group.create_dataset("ID_labels", shape=((np.ceil(set_fractions[set][1]-set_fractions[set][0])*n_imgs), n_IDs), dtype=np.uint8)
            group.create_dataset("emotion_labels", shape=((np.ceil(set_fractions[set][1]-set_fractions[set][0])*n_imgs), n_emotions), dtype=np.uint8)
            group.create_dataset("gender_labels", shape=((np.ceil(set_fractions[set][1]-set_fractions[set][0])*n_imgs), n_genders), dtype=np.uint8)

            for n, filename in enumerate(final_list[img_indices[0]:img_indices[1]]):
                print('\rCreating {} dataset {} %...'.format(set, int(n/set_n_imgs*100)), end='')
                # get image, crop, resize and make grayscale
                image = imread(im_path+'/'+filename)
                dataset_image = np.mean(resize(image[150:150+image.shape[1], :, :], output_shape=[resized_shape, resized_shape, image.shape[2]]), axis=-1, keepdims=True)
                # the ID label is given in the filename
                this_ID_label = int(filename[8:10]) - 1  # -1 to go from [1,73] to [0,72]
                # the emotion label must be created by figuring out which of our requested emotions is in the filename, then assigning a number to each possible emotion
                this_emotion = [em for em in emotions if em in filename]
                this_emotion_label = emotions.index(this_emotion[0])
                # we do something similar as for emotion for the gender
                this_gender = [ge for ge in genders if ge in filename]
                this_gender_label = genders.index(this_gender[0])
                f[set]['data'][n] = dataset_image
                f[set]['ID_labels'][n] = tf.keras.utils.to_categorical(this_ID_label, num_classes=n_IDs)
                f[set]['emotion_labels'][n] = tf.keras.utils.to_categorical(this_emotion_label, num_classes=n_emotions)
                f[set]['gender_labels'][n] = tf.keras.utils.to_categorical(this_gender_label, num_classes=n_genders)


def make_humanbased_hdf5(task, subject=None, resized_shape=224, data_path=r'C:\Users\doeri\PYCHARM_PROJECTS\dfm_dnn\humanbased_DNN_inputs'):
    '''load data from the Radboud Faces Dataset.
       task: str 'emotions' or 'gender'
       subject: int identifier of the subject. If subject=None, the pooled data of all subjects will be used.
       resized_shape: int we will resize images to be squares of this size
       data_path: str path to the file containing the human data
       output_path: str where to save the hdf5 file
       '''

    output_path = './humanbased_dataset_{}_subject_{}.h5'.format(task, subject) if subject is not None else './humanbased_dataset_{}_all_subjects.h5'.format(task)
    if os.path.exists(output_path):
        print('file {} exists. Skipping dataset creation'.format(output_path))
        return

    print('Loading data, this may take a minute.')
    # load .mat containg all data needed
    mat_xp_labels = loadmat(op.join(data_path, f'{task}_INPUT_MAT_DNN.mat'))
    # (subjects x features x labelstype) tensor containing 1) accuracy and 2) actual responses for presented face
    human_classification = mat_xp_labels['human_classification_experiment']
    # (subjects x trials x features presence/absence) tensor containing presence or absence of features at a given trial
    presence_feats_experiment = mat_xp_labels['presence_feats_experiment']
    # (subjects x trials x stimulus presented) tensor indicating which stimulus was presented
    which_stim_experiment = mat_xp_labels['which_stim_experiment']
    # a (stimuli x features x pixels) tensor containing the actual images of gabors.
    mat_img_gabors = h5py.File(op.join(data_path, 'Gabors_vectorized_images.mat'), 'r')
    gabors_images = mat_img_gabors.get('gabor_vectorized_images')
    gabors_images = np.array(gabors_images)

    n_subjects_total = 116
    n_trials_per_subject = 2400
    n_imgs = n_trials_per_subject if subject is not None else n_subjects_total*n_trials_per_subject
    subject_list = [subject] if subject is not None else range(116)
    n_labels = 2
    set_fractions = {'train': [0.0, 0.6], 'val': [0.6, 0.8], 'test': [0.8, 1.0]}  # fraction of all data for each set

    # go through all images, resize to manageable size, get labels and save
    with h5py.File(output_path, "w") as f:
        for set in ["train", "val", "test"]:
            set_indices = [int(set_fractions[set][0]*n_imgs), int(set_fractions[set][1]*n_imgs)]
            set_n_imgs = set_indices[1] - set_indices[0]
            group = f.create_group(set)
            group.create_dataset("n_imgs", data=[set_n_imgs])
            group.create_dataset("data", shape=((np.ceil(set_fractions[set][1]-set_fractions[set][0])*n_imgs), resized_shape, resized_shape, 1), dtype=np.float32)
            group.create_dataset("humanbased_labels", shape=((np.ceil(set_fractions[set][1]-set_fractions[set][0])*n_imgs), n_labels), dtype=np.uint8)

            counter = 0
            for subject in subject_list:
                for trial, index in enumerate(range(set_indices[0], set_indices[1])):
                    print('\rCreating {} dataset {} %...'.format(set, int(counter/set_n_imgs*100)), end='')
                    # get image and label from loaded data
                    this_label = human_classification[subject, trial, 1]  # last dim: 0->trial accuracy, 1->trial response
                    this_stim = int(which_stim_experiment[subject, trial]) - 1  # gets stimulus, -1 changes matlab indexes to python indexes
                    # choose the specific trial, transform in boolean
                    these_gabors = presence_feats_experiment[subject, trial, :] == 1
                    # get all the Gabors images that we presented at this trial, for that stimulus
                    stim_ex = gabors_images[this_stim, these_gabors, :]
                    stim = np.sum(stim_ex, 0)  # sum all Gabs images along first dimension
                    stim = np.reshape(stim, [250, 250])  # reshape the vectorized image to actual 250x250 image
                    stim = np.array(stim).T  # transpose required from matlab to python indexing
                    dataset_image = resize(np.expand_dims(stim, -1), output_shape=[resized_shape, resized_shape, 1])  # add a dimension (needed for convnets) and resize

                    f[set]['data'][trial] = dataset_image
                    f[set]['humanbased_labels'][trial] = tf.keras.utils.to_categorical(this_label, num_classes=n_labels)
                    counter += 1


def check_hdf5_dataset(dataset_type, dataset_path='./RFD_dataset.h5'):
    '''try to read the dataset if you want to check everything is alright
    dataset_type str 'RFD' or 'humanbased'
    '''
    if dataset_type is 'RFD':
        with h5py.File(dataset_path, "r") as f:
            sample_images = f['train']['data'][:25]
            sample_ID_labels = np.argmax(f['train']['ID_labels'][:25], axis=1)  # argmax to go from one_hot to int
            sample_emotion_labels = np.argmax(f['train']['emotion_labels'][:25], axis=1)
            sample_gender_labels = np.argmax(f['train']['gender_labels'][:25], axis=1)
            fig, ax = plt.subplots(5, 5)
            for n in range(25):
                ax[n//5][n%5].imshow(sample_images[n, :, :, 0], cmap='gray')
                ax[n//5][n%5].title.set_text('ID: {}, emotion: {}, gender: {}'.format(sample_ID_labels[n], sample_emotion_labels[n], sample_gender_labels[n]))
    elif dataset_type is 'humanbased':
        with h5py.File(dataset_path, "r") as f:
            sample_images = f['train']['data'][:25]
            sample_labels = np.argmax(f['train']['labels'][:25], axis=1)  # argmax to go from one_hot to int
            fig, ax = plt.subplots(5, 5)
            for n in range(25):
                ax[n//5][n%5].imshow(sample_images[n, :, :, 0], cmap='gray')
                ax[n//5][n%5].title.set_text('Label: {}'.format(sample_labels[n]))
    else:
        raise Exception("dataset type not understood. Please use 'RFD' or 'humanbased'")
    plt.show()


def visualize_batch(batch_images, batch_labels, preds, loss, accuracy):
    preds = tf.argmax(preds, axis=1)
    batch_labels = tf.argmax(batch_labels, axis=1)
    fig, ax = plt.subplots(5, 5)
    for n in range(25):
        ax[n // 5][n % 5].imshow(batch_images[n, :, :, 0], cmap='gray')
        ax[n // 5][n % 5].title.set_text('label: {}, pred {}'.format(batch_labels[n], preds[n]))
    fig.suptitle('batch loss: {}, batch accuracy: {}'.format(loss, accuracy))
    plt.show()


def batch_accuracy(labels, preds):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(preds, axis=1), tf.argmax(labels, axis=1)), tf.float32))


def train_step(model, batch_imgs, batch_labels, loss_fn, optimizer, vars_to_train, visualize_this_batch=False):
    tf.debugging.assert_rank(tf.squeeze(batch_labels), 2, message='please provide one_hot encoded labels')
    with tf.GradientTape() as tape:
        preds = model(batch_imgs)
        accuracy = batch_accuracy(batch_labels, preds)
        loss = loss_fn(batch_labels, preds)
        grad = tape.gradient(loss, vars_to_train)
        optimizer.apply_gradients(zip(grad, vars_to_train))
        if visualize_this_batch: visualize_batch(batch_imgs, batch_labels, preds, loss, accuracy)
    return loss, accuracy


def train_on_hdf5_dataset(model, dataset_path, label_type, batch_size, n_epochs, loss_fn, optimizer, checkpoint, checkpoint_path, saving_manager, augment_data=True, visualize_batches=False):
    '''train and test a model on an hdf5 dataset (see make_RFD_hdf5() for details about the dataset)
    model: a keras model
    dataset_path: str -  the path to the hdf5 dataset
    label_type: str - 'ID_labels', 'emotion_labels' or 'gender_labels', depending on the task you want to use
    the other inputs are self-explanatory'''

    if os.path.exists(checkpoint_path):
        checkpoint.restore(saving_manager.latest_checkpoint)
    else:

        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        augment_seq = iaa.Sequential([iaa.Fliplr(0.5),
                                      iaa.OneOf([iaa.AdditiveGaussianNoise(scale=0.03), iaa.AdditiveLaplaceNoise(scale=0.03), iaa.Dropout(0.03)]),
                                      sometimes(iaa.Affine(scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # scale images to 90-110% of their size, individually per axis
                                                           translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},  # translate by -10 to +10 percent (per axis)
                                                           rotate=(-5, 5),  # rotate by n degrees
                                                           shear=(-5, 5),  # shear by n degrees
                                                          ))
                                     ],
                                     random_order=True)

        with h5py.File(dataset_path, 'r') as f:
            n_train_samples = f['train']['n_imgs'][0]
            n_val_samples = f['val']['n_imgs'][0]
            n_test_samples = f['test']['n_imgs'][0]
            n_train_batches = n_train_samples//batch_size
            n_val_batches = n_val_samples//batch_size
            n_test_batches = n_test_samples//batch_size
            losses = np.zeros(n_train_batches*n_epochs)
            accuracies = np.zeros(n_train_batches*n_epochs)
            val_losses = np.zeros(n_epochs)
            val_accuracies = np.zeros(n_epochs)
            val_counters = np.zeros(n_epochs)  # to catch at which training steps we validated
            test_loss, test_accuracy = 0, 0
            vars_to_train = model.trainable_variables

            for epoch in range(n_epochs):
                indices = np.random.permutation(n_train_samples)  # for shuffling. not finished yet.
                for batch in range(n_train_batches):
                    batch_imgs = f['train']['data'][batch*batch_size:(batch+1)*batch_size]
                    if augment_data: batch_imgs = augment_seq(images=batch_imgs)
                    batch_labels = f['train'][label_type][batch*batch_size:(batch+1)*batch_size]
                    losses[int(checkpoint.step)], accuracies[int(checkpoint.step)] = train_step(model, batch_imgs, batch_labels, loss_fn, optimizer, vars_to_train, visualize_batches)
                    if int(checkpoint.step) % 25 == 0:
                        print('\rTraining - Epoch {}, batch {} -- loss = {}, accuracy = {} || Last valid - Epoch {} -- loss = {}, accuracy = {}'.format(epoch, batch, losses[int(checkpoint.step)], accuracies[int(checkpoint.step)], epoch-1, val_losses[max(epoch-1,0)], val_accuracies[max(epoch-1,0)]), end=' ')
                    checkpoint.step.assign_add(1)
                    if int(checkpoint.step) % 2500 == 0 or int(checkpoint.step) == (n_train_batches)*(n_epochs):
                        save_path = saving_manager.save()
                        print("\nSaved checkpoints for step {}: {} (epoch {}).".format(int(checkpoint.step), save_path, epoch))

                for batch in range(n_val_batches):
                    batch_imgs = f['val']['data'][batch * batch_size:(batch + 1) * batch_size]
                    batch_labels = f['val'][label_type][batch * batch_size:(batch + 1) * batch_size]
                    val_losses[epoch] += loss_fn(batch_labels, model(batch_imgs))/n_val_batches
                    val_accuracies[epoch] += batch_accuracy(batch_labels, model(batch_imgs))/n_val_batches
                    val_counters[epoch] = int(checkpoint.step)

            print('\nComputing performance on test set...')
            for batch in range(n_test_batches):
                batch_imgs = f['test']['data'][batch * batch_size:(batch + 1) * batch_size]
                batch_labels = f['test'][label_type][batch * batch_size:(batch + 1) * batch_size]
                test_loss += loss_fn(batch_labels, model(batch_imgs)) / n_test_batches
                test_accuracy += batch_accuracy(batch_labels, model(batch_imgs)) / n_test_batches
                if batch % 25 == 0 or batch == n_train_batches-1:
                    print('\rTesting progress: {} %'.format(batch / n_test_batches * 100), end=' ')
            print('\nTesting loss = {}\nTesting accuracy = {}'.format(test_loss, test_accuracy))

            fig, a = plt.subplots(1, 2)
            a[0].plot(range(n_train_batches*n_epochs), losses, label='train')
            a[0].plot(val_counters, val_losses, 'r', label='val')
            a[0].plot(n_train_batches*n_epochs, test_loss, 'go', label='test')
            a[0].legend(loc='upper right')
            a[1].plot(range(n_train_batches*n_epochs), accuracies)
            a[1].plot(val_counters, val_accuracies, 'r')
            a[1].plot(n_train_batches*n_epochs, test_accuracy, 'go')
            plt.savefig(checkpoint_path[:-4]+'learning_curves.png')

        return losses, accuracies
