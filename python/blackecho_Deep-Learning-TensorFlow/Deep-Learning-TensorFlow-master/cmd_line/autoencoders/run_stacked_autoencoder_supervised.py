import numpy as np
import tensorflow as tf

from yadlt.models.autoencoders import stacked_denoising_autoencoder
from yadlt.utils import datasets, utilities

# #################### #
#   Flags definition   #
# #################### #
flags = tf.app.flags
FLAGS = flags.FLAGS

# Global configuration
flags.DEFINE_string('dataset', 'mnist', 'Which dataset to use. ["mnist", "cifar10", "custom"]')
flags.DEFINE_string('train_dataset', '', 'Path to train set .npy file.')
flags.DEFINE_string('train_labels', '', 'Path to train labels .npy file.')
flags.DEFINE_string('valid_dataset', '', 'Path to valid set .npy file.')
flags.DEFINE_string('valid_labels', '', 'Path to valid labels .npy file.')
flags.DEFINE_string('test_dataset', '', 'Path to test set .npy file.')
flags.DEFINE_string('test_labels', '', 'Path to test labels .npy file.')
flags.DEFINE_string('cifar_dir', '', 'Path to the cifar 10 dataset directory.')
flags.DEFINE_boolean('do_pretrain', True, 'Whether or not doing unsupervised pretraining.')
flags.DEFINE_string('save_predictions', '', 'Path to a .npy file to save predictions of the model.')
flags.DEFINE_string('save_layers_output_test', '', 'Path to a .npy file to save test set output from all the layers of the model.')
flags.DEFINE_string('save_layers_output_train', '', 'Path to a .npy file to save train set output from all the layers of the model.')
flags.DEFINE_integer('seed', -1, 'Seed for the random generators (>= 0). Useful for testing hyperparameters.')
flags.DEFINE_string('name', 'sdae', 'Name for the model.')
flags.DEFINE_float('momentum', 0.5, 'Momentum parameter.')

# Supervised fine tuning parameters
flags.DEFINE_string('finetune_loss_func', 'softmax_cross_entropy', 'Last Layer Loss function. ["softmax_cross_entropy", "mse"]')
flags.DEFINE_integer('finetune_num_epochs', 30, 'Number of epochs for the fine-tuning phase.')
flags.DEFINE_float('finetune_learning_rate', 0.001, 'Learning rate for the fine-tuning phase.')
flags.DEFINE_string('finetune_act_func', 'relu', 'Activation function for the fine-tuning phase. ["sigmoid, "tanh", "relu"]')
flags.DEFINE_float('finetune_dropout', 1, 'Dropout parameter.')
flags.DEFINE_string('finetune_opt', 'sgd', '["sgd", "adagrad", "momentum", "adam"]')
flags.DEFINE_integer('finetune_batch_size', 20, 'Size of each mini-batch for the fine-tuning phase.')
# Autoencoder layers specific parameters
flags.DEFINE_string('dae_layers', '256,', 'Comma-separated values for the layers in the sdae.')
flags.DEFINE_string('dae_regcoef', '5e-4,', 'Regularization parameter for the autoencoders. If 0, no regularization.')
flags.DEFINE_string('dae_enc_act_func', 'sigmoid,', 'Activation function for the encoder. ["sigmoid", "tanh"]')
flags.DEFINE_string('dae_dec_act_func', 'none,', 'Activation function for the decoder. ["sigmoid", "tanh", "none"]')
flags.DEFINE_string('dae_loss_func', 'mse,', 'Loss function. ["mse" or "cross_entropy"]')
flags.DEFINE_string('dae_opt', 'sgd,', '["sgd", "ada_grad", "momentum", "adam"]')
flags.DEFINE_string('dae_learning_rate', '0.01,', 'Initial learning rate.')
flags.DEFINE_string('dae_num_epochs', '10,', 'Number of epochs.')
flags.DEFINE_string('dae_batch_size', '10,', 'Size of each mini-batch.')
flags.DEFINE_string('dae_corr_type', 'none,', 'Type of input corruption. ["none", "masking", "salt_and_pepper"]')
flags.DEFINE_string('dae_corr_frac', '0.0,', 'Fraction of the input to corrupt.')

# Conversion of Autoencoder layers parameters from string to their specific type
dae_layers = utilities.flag_to_list(FLAGS.dae_layers, 'int')
dae_enc_act_func = utilities.flag_to_list(FLAGS.dae_enc_act_func, 'str')
dae_dec_act_func = utilities.flag_to_list(FLAGS.dae_dec_act_func, 'str')
dae_opt = utilities.flag_to_list(FLAGS.dae_opt, 'str')
dae_loss_func = utilities.flag_to_list(FLAGS.dae_loss_func, 'str')
dae_learning_rate = utilities.flag_to_list(FLAGS.dae_learning_rate, 'float')
dae_regcoef = utilities.flag_to_list(FLAGS.dae_regcoef, 'float')
dae_corr_type = utilities.flag_to_list(FLAGS.dae_corr_type, 'str')
dae_corr_frac = utilities.flag_to_list(FLAGS.dae_corr_frac, 'float')
dae_num_epochs = utilities.flag_to_list(FLAGS.dae_num_epochs, 'int')
dae_batch_size = utilities.flag_to_list(FLAGS.dae_batch_size, 'int')

# Parameters validation
assert all([0. <= cf <= 1. for cf in dae_corr_frac])
assert all([ct in ['masking', 'salt_and_pepper', 'none'] for ct in dae_corr_type])
assert FLAGS.dataset in ['mnist', 'cifar10', 'custom']
assert len(dae_layers) > 0
assert all([af in ['sigmoid', 'tanh'] for af in dae_enc_act_func])
assert all([af in ['sigmoid', 'tanh', 'none'] for af in dae_dec_act_func])

if __name__ == '__main__':

    utilities.random_seed_np_tf(FLAGS.seed)

    if FLAGS.dataset == 'mnist':

        # ################# #
        #   MNIST Dataset   #
        # ################# #

        trX, trY, vlX, vlY, teX, teY = datasets.load_mnist_dataset(mode='supervised')

    elif FLAGS.dataset == 'cifar10':

        # ################### #
        #   Cifar10 Dataset   #
        # ################### #

        trX, trY, teX, teY = datasets.load_cifar10_dataset(FLAGS.cifar_dir, mode='supervised')
        # Validation set is the first half of the test set
        vlX = teX[:5000]
        vlY = teY[:5000]

    elif FLAGS.dataset == 'custom':

        # ################## #
        #   Custom Dataset   #
        # ################## #

        def load_from_np(dataset_path):
            if dataset_path != '':
                return np.load(dataset_path)
            else:
                return None

        trX, trY = load_from_np(FLAGS.train_dataset), load_from_np(FLAGS.train_labels)
        vlX, vlY = load_from_np(FLAGS.valid_dataset), load_from_np(FLAGS.valid_labels)
        teX, teY = load_from_np(FLAGS.test_dataset), load_from_np(FLAGS.test_labels)

    else:
        trX = None
        trY = None
        vlX = None
        vlY = None
        teX = None
        teY = None

    # Create the object
    sdae = None

    dae_enc_act_func = [utilities.str2actfunc(af) for af in dae_enc_act_func]
    dae_dec_act_func = [utilities.str2actfunc(af) for af in dae_dec_act_func]
    finetune_act_func = utilities.str2actfunc(FLAGS.finetune_act_func)

    sdae = stacked_denoising_autoencoder.StackedDenoisingAutoencoder(
        do_pretrain=FLAGS.do_pretrain, name=FLAGS.name,
        layers=dae_layers, finetune_loss_func=FLAGS.finetune_loss_func,
        finetune_learning_rate=FLAGS.finetune_learning_rate, finetune_num_epochs=FLAGS.finetune_num_epochs,
        finetune_opt=FLAGS.finetune_opt, finetune_batch_size=FLAGS.finetune_batch_size,
        finetune_dropout=FLAGS.finetune_dropout,
        enc_act_func=dae_enc_act_func, dec_act_func=dae_dec_act_func,
        corr_type=dae_corr_type, corr_frac=dae_corr_frac, regcoef=dae_regcoef,
        loss_func=dae_loss_func, opt=dae_opt,
        learning_rate=dae_learning_rate, momentum=FLAGS.momentum,
        num_epochs=dae_num_epochs, batch_size=dae_batch_size,
        finetune_act_func=finetune_act_func)

    # Fit the model (unsupervised pretraining)
    if FLAGS.do_pretrain:
        encoded_X, encoded_vX = sdae.pretrain(trX, vlX)

    # Supervised finetuning
    sdae.fit(trX, trY, vlX, vlY)

    # Compute the accuracy of the model
    print('Test set accuracy: {}'.format(sdae.score(teX, teY)))

    # Save the predictions of the model
    if FLAGS.save_predictions:
        print('Saving the predictions for the test set...')
        np.save(FLAGS.save_predictions, sdae.predict(teX))

    def save_layers_output(which_set):
        if which_set == 'train':
            trout = sdae.get_layers_output(trX)
            for i, o in enumerate(trout):
                np.save(FLAGS.save_layers_output_train + '-layer-' + str(i + 1) + '-train', o)

        elif which_set == 'test':
            teout = sdae.get_layers_output(teX)
            for i, o in enumerate(teout):
                np.save(FLAGS.save_layers_output_test + '-layer-' + str(i + 1) + '-test', o)

    # Save output from each layer of the model
    if FLAGS.save_layers_output_test:
        print('Saving the output of each layer for the test set')
        save_layers_output('test')

    # Save output from each layer of the model
    if FLAGS.save_layers_output_train:
        print('Saving the output of each layer for the train set')
        save_layers_output('train')
