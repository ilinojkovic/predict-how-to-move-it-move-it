import tensorflow as tf
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import RNNCell


class ResidualWrapper(RNNCell):
    """Operator adding residual connections to a given cell."""

    def __init__(self, cell):
        """Create a cell with added residual connection.
        Args:
          cell: an RNNCell. The input is added to the output.
        Raises:
          TypeError: if cell is not an RNNCell.
        """
        if not isinstance(cell, RNNCell):
            raise TypeError("The parameter cell is not a RNNCell.")

        self._cell = cell

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        """Run the cell and add a residual connection."""

        # Run the rnn as usual
        output, new_state = self._cell(inputs, state, scope)

        # Add the residual connection
        output = tf.add(output, inputs[:, :tf.shape(output)[1]])

        return output, new_state


class LinearSpaceDecoderWrapper(RNNCell):
    """Operator adding a linear encoder to an RNN cell"""

    def __init__(self, cell, output_size):
        """Create a cell with with a linear encoder in space.
        Args:
          cell: an RNNCell. The input is passed through a linear layer.
        Raises:
          TypeError: if cell is not an RNNCell.
        """
        if not isinstance(cell, RNNCell):
            raise TypeError("The parameter cell is not a RNNCell.")

        self._cell = cell

        print('output_size = {0}'.format(output_size))
        print(' state_size = {0}'.format(self._cell.state_size))

        # Tuple if multi-rnn
        if isinstance(self._cell.state_size, tuple):

            # Fine if GRU...
            insize = self._cell.state_size[-1]

            # LSTMStateTuple if LSTM
            # if isinstance(insize, LSTMStateTuple):
            #     insize = insize.h

        else:
            # Fine if not multi-rnn
            insize = self._cell.state_size

        self.w_out = tf.get_variable("proj_w_out",
                                     [insize, output_size],
                                     dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))
        self.b_out = tf.get_variable("proj_b_out", [output_size],
                                     dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))

        self.linear_output_size = output_size

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self.linear_output_size

    def __call__(self, inputs, state, scope=None):
        """Use a linear layer and pass the output to the cell."""

        # Run the rnn as usual
        output, new_state = self._cell(inputs, state, scope)

        # Apply the multiplication to everything
        output = tf.matmul(output, self.w_out) + self.b_out

        return output, new_state


class RNNModel(object):
    """
    Creates training and validation computational graphs.
    Note that tf.variable_scope enables parameter sharing so that both graphs are identical.
    """

    def __init__(self, config, placeholders, mode):
        """
        Basic setup.
        :param config: configuration dictionary
        :param placeholders: dictionary of input placeholders
        :param mode: training, validation or inference
        """
        assert mode in ['training', 'validation', 'inference']
        self.config = config
        self.encoder_seq_len = config['encoder_seq_len']
        self.decoder_seq_len = config['decoder_seq_len']
        self.encoder_input_raw = placeholders['enc_in_pl']
        self.decoder_input_raw = placeholders['dec_in_pl']
        self.decoder_target_raw = placeholders['dec_out_pl']
        self.action_labels = placeholders['action_labels_pl']
        self.mask = placeholders['mask_pl']
        self.mode = mode
        self.is_training = self.mode == 'training'
        self.reuse = self.mode == 'validation'
        self.input_dim = config['input_dim']
        self.output_dim = config['output_dim']
        self.summary_collection = 'training_summaries' if mode == 'training' else 'validation_summaries'
        self.hidden_state_size = config['hidden_state_size']
        self.num_layers = config['num_layers']

        # === Transform the inputs ===
        with tf.name_scope('input'):
            self.action_one_hot = tf.one_hot(self.action_labels, self.config['num_actions'])

            self.encoder_input_ = tf.transpose(self.encoder_input_raw, [1, 0, 2])
            self.decoder_input_ = tf.transpose(self.decoder_input_raw, [1, 0, 2])
            self.decoder_target = tf.transpose(self.decoder_target_raw, [1, 0, 2])

            self.encoder_input_ = tf.reshape(self.encoder_input_, [-1, self.input_dim])
            self.decoder_input_ = tf.reshape(self.decoder_input_, [-1, self.input_dim])
            self.decoder_target = tf.reshape(self.decoder_target, [-1, self.input_dim])

            self.encoder_input_ = tf.split(self.encoder_input_, self.encoder_seq_len - 1, axis=0)
            self.decoder_input_ = tf.split(self.decoder_input_, self.decoder_seq_len, axis=0)
            self.decoder_target = tf.split(self.decoder_target, self.decoder_seq_len, axis=0)

            self.encoder_input_ = [tf.concat([enc_input, self.action_one_hot], axis=1) for enc_input in self.encoder_input_]
            self.decoder_input_ = [tf.concat([dec_input, self.action_one_hot], axis=1) for dec_input in self.decoder_input_]

    def build_graph(self):
        self.build_model()
        self.build_loss()
        self.count_parameters()

    def build_model(self):
        """
        Builds the actual model.
        """
        # TODO Implement your model here
        # Some hints:
        #   1) You can access an input batch via `self.input_` and the corresponding targets via `self.target`. Note
        #      that the shape of each input and target is (batch_size, max_seq_length, input_dim)
        #
        #   2) The sequence length of each batch entry is variable, i.e. one entry in the batch might have length
        #      99 while another has length 67. No entry will be larger than what you supplied in
        #      `self.config['max_seq_length']`. This maximum sequence length is also available via `self.max_seq_length`
        #      Because TensorFlow cannot handle variable length sequences out-of-the-box, the data loader pads all
        #      batch entries with zeros so that they have size `self.max_seq_length`. The true sequence lengths are
        #      stored in `self.seq_lengths`. Furthermore, `self.mask` is a mask of shape
        #      `(batch_size, self.max_seq_length)` whose entries are 0 if this entry was padded and 1 otherwise.
        #
        #   3) You can access the config via `self.config`
        #
        #   4) The following member variables should be set after you complete this part:
        #      - `self.initial_state`: a reference to the initial state of the RNN
        #      - `self.final_state`: the final state of the RNN after the outputs have been obtained
        #      - `self.prediction`: the actual output of the model in shape `(batch_size, self.max_seq_length, output_dim)`

        with tf.variable_scope('rnn_model', reuse=self.reuse):
            # Martinez seq2seq
            cells = [tf.contrib.rnn.GRUCell(self.hidden_state_size) for _ in range(self.num_layers)]
            cell = tf.contrib.rnn.MultiRNNCell(cells)

            # Add space decoder
            cell = LinearSpaceDecoderWrapper(cell, self.input_dim)

            # Finally, wrap everything in a residual layer if we want to model velocities
            cell = ResidualWrapper(cell)

            def lf(prev, i):  # function for sampling_based loss
                return tf.concat([prev, self.action_one_hot], axis=1)

            self.outputs, self.final_state = tf.contrib.legacy_seq2seq.tied_rnn_seq2seq(self.encoder_input_,
                                                                                        self.decoder_input_,
                                                                                        cell,
                                                                                        loop_function=lf)
            stacked_outputs = tf.stack(self.outputs)
            self.prediction = tf.transpose(stacked_outputs, [1, 0, 2])

    def build_loss(self):
        """
        Builds the loss function.
        """
        # only need loss if we are not in inference mode
        if self.mode is not 'inference':
            with tf.name_scope('loss'):
                # TODO Implement your loss here
                # You can access the outputs of the model via `self.prediction` and the corresponding targets via
                # `self.target`. Hint 1: you will want to use the provided `self.mask` to make sure that padded values
                # do not influence the loss. Hint 2: L2 loss is probably a good starting point ...

                decoder_mask = self.mask[:, self.encoder_seq_len:]
                expanded_mask = tf.expand_dims(decoder_mask, axis=-1)
                expanded_mask = tf.transpose(expanded_mask, [1, 0, 2])
                self.loss = tf.losses.mean_squared_error(labels=self.decoder_target, predictions=self.outputs,
                                                         weights=expanded_mask)

                tf.summary.scalar('loss', self.loss, collections=[self.summary_collection])

    def count_parameters(self):
        """
        Counts the number of trainable parameters in this model
        """

        self.n_parameters = 0
        for v in tf.trainable_variables():
            params = 1
            for s in v.get_shape():
                params *= s.value
            self.n_parameters += params

    def get_feed_dict(self, batch):
        """
        Returns the feed dictionary required to run one training step with the model.
        :param batch: The mini batch of data to feed into the model
        :return: A feed dict that can be passed to a session.run call
        """
        input_padded, target_padded = batch.get_padded_data()
        encoder_input = input_padded[:, :self.encoder_seq_len - 1, :]
        decoder_input = input_padded[:, self.encoder_seq_len - 1: self.encoder_seq_len + self.decoder_seq_len - 1, :]
        decoder_target = target_padded[:, self.encoder_seq_len:, :]
        feed_dict = {self.encoder_input_raw: encoder_input,
                     self.decoder_input_raw: decoder_input,
                     self.decoder_target_raw: decoder_target,
                     self.action_labels: batch.action_labels,
                     self.mask: batch.mask}

        return feed_dict
