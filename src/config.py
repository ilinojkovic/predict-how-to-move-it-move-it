# configuration used by the training and evaluation scripts
train_config = {}
train_config['data_dir'] = '../data'  # TODO where the data downloaded from Kaggle is stored, i.e. the *.npz files
train_config['output_dir'] = '../trained_models/'  # TODO where you want to store the checkpoints of different training runs
train_config['name'] = 'a_name'  # TODO give your model a name if you want
train_config['batch_size'] = 64  # TODO specify a batch size
train_config['max_seq_length'] = 75  # TODO specify for how many time steps you want to unroll the RNN
train_config['encoder_seq_len'] = 50
train_config['decoder_seq_len'] = 25
train_config['num_actions'] = 15

train_config['hidden_state_size'] = 1024  # TODO specify the hidden state size
train_config['num_layers'] = 1  # TODO Specify the number of layers

train_config['learning_rate'] = 0.005  # TODO specify a learning rate (this is currently just a dummy value)
train_config['n_epochs'] = 5  # TODO for how many epochs to train (this is currently just a dummy value)
train_config['save_checkpoints_every_epoch'] = 1  # after how many epochs the trained model should be saved
train_config['n_keep_checkpoints'] = 1  # how many saved checkpoints to keep
train_config['gradient_clip'] = 5  # TODO gradient clipping scalar

# some code to anneal the learning rate, this is implemented for you, you can just choose it here
train_config['learning_rate_type'] = 'fixed'  # ['fixed', 'exponential', 'linear']
train_config['learning_rate_decay_steps'] = 1000
train_config['learning_rate_decay_rate'] = 0.95

# TODO add more configurations to your liking, e.g. type activation functions, type of optimizer, various model parameters etc.

# some additional configuration parameters required when the configured model is used at inference time
test_config = train_config.copy()
test_config['max_seq_length'] = -1  # want to use entire sequence during test, which is fixed to 50, don't change this
test_config['model_dir'] = '../trained_models/a_name_1527448279/'  # TODO path to the model that you want to evaluate
test_config['checkpoint_id'] = None  # if None, the last checkpoint will be used
test_config['prediction_length'] = 25  # how many frames to predict into the future (assignment requires 25 frames, but you can experiment with more if you'd like)
