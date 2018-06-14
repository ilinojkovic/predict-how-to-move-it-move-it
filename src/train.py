import datetime
import os
import tensorflow as tf
import time
import sys

from config import train_config

from model import Seq2SeqModel
from load_data import MotionDataset
from utils import export_config, calculate_stats, save_stats, preprocess


def load_data(config, split, stride):
    print('Loading data from {} ...'.format(config['data_dir']))
    return MotionDataset.load(data_path=config['data_dir'],
                              split=split,
                              seq_length=config['max_seq_length'],
                              batch_size=config['batch_size'],
                              stride=stride)


def get_model_and_placeholders(config):
    # create placeholders that we need to feed the required data into the model
    # None means that the dimension is variable, which we want for the batch size and the sequence length
    input_dim = output_dim = config['input_dim']

    with tf.name_scope('Input'):
        enc_in_pl = tf.placeholder(tf.float32, shape=[None, config['encoder_seq_len'] - 1, input_dim], name="enc_in")
        dec_in_pl = tf.placeholder(tf.float32, shape=[None, config['decoder_seq_len'], input_dim], name="dec_in")
        dec_out_pl = tf.placeholder(tf.float32, shape=[None, config['decoder_seq_len'], output_dim], name="dec_out")
        action_labels_pl = tf.placeholder(tf.int32, shape=[None], name='action_labels')
        mask_pl = tf.placeholder(tf.float32, shape=[None, None], name='mask_pl')

    placeholders = {'enc_in_pl': enc_in_pl,
                    'dec_in_pl': dec_in_pl,
                    'dec_out_pl': dec_out_pl,
                    'action_labels_pl': action_labels_pl,
                    'mask_pl': mask_pl}

    seq2seq_model_class = Seq2SeqModel
    return seq2seq_model_class, placeholders


def main(config):
    # create unique output directory for this model

    config['name'] = config['name'] + '-' + str(config['hidden_state_size'])
    if config['train_stride']:
        config['name'] = config['name'] + '-stride'
    if config['concat_labels']:
        config['name'] = config['name'] + '-concat_labels'
    if config['attention']:
        config['name'] = config['name'] + '-attention'
    if config['share_weights']:
        config['name'] = config['name'] + '-share_weights'

    config['name'] = config['name'] + '-' + config['learning_rate_type'] + '-' + str(config['learning_rate'])

    timestamp = str(int(time.time()))
    config['model_dir'] = os.path.abspath(os.path.join(config['output_dir'], config['name'] + '-' + timestamp))
    os.makedirs(config['model_dir'])
    print('Writing checkpoints into {}'.format(config['model_dir']))

    # load the data, this requires that the *.npz files you downloaded from Kaggle be named `train.npz` and `valid.npz`
    data_train = load_data(config, 'train', config['train_stride'])
    data_valid = load_data(config, 'valid', config['eval_stride'])

    # TODO if you would like to do any preprocessing of the data, here would be a good opportunity
    stats = calculate_stats(data_train.input_)
    save_stats(stats)

    if config['normalize']:
        data_train.input_, _, _ = preprocess(data_train.input_)
        data_train.target, _, _ = preprocess(data_train.target)

        data_valid.input_, _, _ = preprocess(data_valid.input_)
        data_valid.target, _, _ = preprocess(data_valid.target)

        print('Post normalize samples shape: ', data_train.input_[0].shape)

    config['input_dim'] = data_train.input_[0].shape[-1]
    config['output_dim'] = data_train.target[0].shape[-1]

    # get input placeholders and get the model that we want to train
    seq2seq_model_class, placeholders = get_model_and_placeholders(config)

    # Create a variable that stores how many training iterations we performed.
    # This is useful for saving/storing the network
    global_step = tf.Variable(1, name='global_step', trainable=False)

    # create a training graph, this is the graph we will use to optimize the parameters
    with tf.name_scope('Training'):
        seq2seq_model = seq2seq_model_class(config, placeholders, mode='training')
        seq2seq_model.build_graph()
        print('created RNN model with {} parameters'.format(seq2seq_model.n_parameters))

        # configure learning rate
        if config['learning_rate_type'] == 'exponential':
            lr = tf.train.exponential_decay(config['learning_rate'],
                                            global_step=global_step,
                                            decay_steps=config['learning_rate_decay_steps'],
                                            decay_rate=config['learning_rate_decay_rate'],
                                            staircase=False)
            lr_decay_op = tf.identity(lr)
        elif config['learning_rate_type'] == 'linear':
            lr = tf.Variable(config['learning_rate'], trainable=False)
            lr_decay_op = lr.assign(tf.multiply(lr, config['learning_rate_decay_rate']))
        elif config['learning_rate_type'] == 'fixed':
            lr = config['learning_rate']
            lr_decay_op = tf.identity(lr)
        else:
            raise ValueError('learning rate type "{}" unknown.'.format(config['learning_rate_type']))

        with tf.name_scope('Step'):
            # TODO choose the optimizer you desire here and define `train_op. The loss should be accessible through rnn_model.loss
            params = tf.trainable_variables()
            optimizer = tf.train.AdamOptimizer(config['learning_rate'])
            gradients = tf.gradients(seq2seq_model.loss, params)

            # clip the gradients to counter explosion
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, config['gradient_clip'])

            # backprop
            train_op = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)

    # create a graph for validation
    with tf.name_scope('Validation'):
        seq2seq_model_valid = seq2seq_model_class(config, placeholders, mode='validation')
        seq2seq_model_valid.build_graph()

    # Create summary ops for monitoring the training
    # Each summary op annotates a node in the computational graph and collects data data from it
    tf.summary.scalar('learning_rate', lr, collections=['training_summaries'])

    # Merge summaries used during training and reported after every step
    summaries_training = tf.summary.merge(tf.get_collection('training_summaries'))

    # create summary ops for monitoring the validation
    # caveat: we want to store the performance on the entire validation set, not just one validation batch
    # Tensorflow does not directly support this, so we must process every batch independently and then aggregate
    # the results outside of the model
    # so, we create a placeholder where can feed the aggregated result back into the model
    loss_valid_pl = tf.placeholder(tf.float32, name='loss_valid_pl')
    loss_valid_s = tf.summary.scalar('loss_valid', loss_valid_pl, collections=['validation_summaries'])

    # merge validation summaries
    summaries_valid = tf.summary.merge([loss_valid_s])

    # dump the config to the model directory in case we later want to see it
    export_config(config, os.path.join(config['model_dir'], 'config.txt'))

    with tf.Session() as sess:
        # Add the ops to initialize variables.
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        # Actually intialize the variables
        sess.run(init_op)

        # create file writers to dump summaries onto disk so that we can look at them with tensorboard
        train_summary_dir = os.path.join(config['model_dir'], "summary", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        valid_summary_dir = os.path.join(config['model_dir'], "summary", "validation")
        valid_summary_writer = tf.summary.FileWriter(valid_summary_dir, sess.graph)

        # create a saver for writing training checkpoints
        saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=config['n_keep_checkpoints'])

        # start training
        start_time = time.time()
        current_step = 0
        for e in range(config['n_epochs']):

            # reshuffle the batches
            data_train.reshuffle()

            # loop through all training batches
            for i, batch in enumerate(data_train.all_batches()):
                step = tf.train.global_step(sess, global_step)
                current_step += 1

                if config['learning_rate_type'] == 'linear' and current_step % config['learning_rate_decay_steps'] == 0:
                    sess.run(lr_decay_op)

                # we want to train, so must request at least the train_op
                fetches = {'summaries': summaries_training,
                           'loss': seq2seq_model.loss,
                           'train_op': train_op}

                # get the feed dict for the current batch
                feed_dict = seq2seq_model.get_feed_dict(batch)

                # feed data into the model and run optimization
                training_out = sess.run(fetches, feed_dict)

                # write logs
                train_summary_writer.add_summary(training_out['summaries'], global_step=step)

                # print training performance of this batch onto console
                time_delta = str(datetime.timedelta(seconds=int(time.time() - start_time)))
                print('\rEpoch: {:3d} [{:4d}/{:4d}] time: {:>8} loss: {:.4f}'.format(
                    e + 1, i + 1, data_train.n_batches, time_delta, training_out['loss']), end='')

            # after every epoch evaluate the performance on the validation set
            total_valid_loss = 0.0
            n_valid_samples = 0
            for batch in data_valid.all_batches():
                fetches = {'loss': seq2seq_model_valid.loss}
                feed_dict = seq2seq_model_valid.get_feed_dict(batch)
                valid_out = sess.run(fetches, feed_dict)

                total_valid_loss += valid_out['loss'] * batch.batch_size
                n_valid_samples += batch.batch_size

            # write validation logs
            avg_valid_loss = total_valid_loss / n_valid_samples
            valid_summaries = sess.run(summaries_valid, {loss_valid_pl: avg_valid_loss})
            valid_summary_writer.add_summary(valid_summaries, global_step=tf.train.global_step(sess, global_step))

            # print validation performance onto console
            print(' | validation loss: {:.6f}'.format(avg_valid_loss))

            # save this checkpoint if necessary
            if (e + 1) % config['save_checkpoints_every_epoch'] == 0:
                saver.save(sess, os.path.join(config['model_dir'], 'model'), global_step)

            if avg_valid_loss > 20:
                break

        # Training finished, always save model before exiting
        print('Training finished')
        ckpt_path = saver.save(sess, os.path.join(config['model_dir'], 'model'), global_step)
        print('Model saved to file {}'.format(ckpt_path))


if __name__ == '__main__':

    print('Running model: ', sys.argv)

    train_config['train_stride'] = bool(int(sys.argv[1]))
    train_config['concat_labels'] = bool(int(sys.argv[2]))
    train_config['hidden_state_size'] = int(sys.argv[3])
    train_config['attention'] = bool(int(sys.argv[4]))
    train_config['share_weights'] = bool(int(sys.argv[5]))
    train_config['learning_rate_type'] = sys.argv[6]

    assert train_config['learning_rate_type'] in ['exponential', 'fixed']

    if train_config['learning_rate_type'] == 'exponential':
        train_config['learning_rate'] = 0.005
    else:
        train_config['learning_rate'] = 0.0001

    main(train_config)
