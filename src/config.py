# configuration used by the training and evaluation scripts
train_config = {}

#####
# Select name, data location and output location
#####
train_config['data_dir'] = '../data'  # TODO where the data downloaded from Kaggle is stored, i.e. the *.npz files
train_config['output_dir'] = '../trained_models/parameter_tuning'  # TODO where you want to store the checkpoints of different training runs
train_config['name'] = 'model'  # TODO give your model a name if you want

#####
# Select model parameters
#####
train_config['hidden_state_size'] = 128  # TODO Specify the hidden state size
train_config['num_layers'] = 1  # TODO Specify the number of layers
train_config['max_seq_length'] = 75  # TODO specify for how many time steps you want to unroll the RNN
train_config['encoder_seq_len'] = 50
train_config['decoder_seq_len'] = 25
train_config['share_weights'] = True  # TODO specify whether the encoder and the decoder should share weights or not
train_config['attention'] = False  # TODO specify whether attention should be used or not
train_config['dropout'] = False  # TODO Specify whether dropout of 0.5 should be used or not
train_config['normalize'] = False  # TODO specify whether data should be min-max normalized before training
train_config['train_stride'] = True  # TODO specify whether striding or splitting should be used during data loading
train_config['eval_stride'] = False  # TODO specify whether striding should be used for validation and test
train_config['stride_value'] = 25  # TODO specify shifting size for striding
train_config['model_velocities'] = True  # TODO specify whether velocities should be modeled
train_config['concat_labels'] = False  # TODO specify whether class labels should be concatenated to the angles
train_config['num_actions'] = 15  # TODO specify the amount of different labels

#####
# Select optimization and training parameters
#####
train_config['batch_size'] = 16  # TODO specify a batch size
train_config['learning_rate'] = 0.0001  # TODO specify a learning rate (this is currently just a dummy value
train_config['n_epochs'] = 150  # TODO for how many epochs to train (this is currently just a dummy value)
train_config['save_checkpoints_every_epoch'] = 5  # TODO after how many epochs the trained model should be saved
train_config['n_keep_checkpoints'] = 30  # TODO how many saved checkpoints to keep
train_config['gradient_clip'] = 5  # TODO gradient clipping scalar
train_config['learning_rate_type'] = 'fixed'  # TODO choose between ['fixed', 'exponential', 'linear']
train_config['learning_rate_decay_steps'] = 1000
train_config['learning_rate_decay_rate'] = 0.95

# some additional configuration parameters required when the configured model is used at inference time
test_config = train_config.copy()
test_config['max_seq_length'] = -1  # want to use entire sequence during test, which is fixed to 50, don't change this
test_config['model_dir'] = '../trained_models/a_name_1528971734/'  # TODO path to the model that you want to evaluate
test_config['checkpoint_id'] = 5221  # if None, the last checkpoint will be used
test_config['prediction_length'] = 25  # how many frames to predict into the future (assignment requires 25 frames, but you can experiment with more if you'd like)
