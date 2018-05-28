import tensorflow as tf


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
        self.input_ = placeholders['input_pl']
        self.encoder_input_ = self.input_[:, :50, :]
        self.decoder_input_ = self.input_[:, 50:, :]
        self.target = placeholders['target_pl']
        self.mask = placeholders['mask_pl']
        self.seq_lengths = placeholders['seq_lengths_pl']
        self.encoder_seq_lengths = [50] * tf.shape(self.seq_lengths)[0]
        self.decoder_seq_lengths = [25] * tf.shape(self.seq_lengths)[0]
        self.mode = mode
        self.is_training = self.mode == 'training'
        self.reuse = self.mode == 'validation'
        self.batch_size = tf.shape(self.input_)[0]  # dynamic size
        self.max_seq_length = tf.shape(self.input_)[1]  # dynamic size
        self.input_dim = config['input_dim']
        self.output_dim = config['output_dim']
        self.summary_collection = 'training_summaries' if mode == 'training' else 'validation_summaries'
        self.hidden_state_size = config['hidden_state_size']
        self.num_layers = config['num_layers']

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

            # cells = [tf.contrib.rnn.GRUCell(num_units=self.hidden_state_size) for _ in range(self.num_layers)]
            #
            # # we stack the cells together and create one big RNN cell
            # cell = tf.contrib.rnn.MultiRNNCell(cells)
            #
            # self.initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)
            #
            # outputs, self.final_state = tf.nn.dynamic_rnn(cell=cell, initial_state=self.initial_state,
            #                                               inputs=self.input_, sequence_length=self.seq_lengths)
            #
            # local_max_seq_length = tf.shape(self.input_)[1]

            # outputs_flat = tf.reshape(outputs, [-1, self.hidden_state_size])
            # down_project = tf.layers.dense(outputs_flat, units=self.output_dim)
            # self.prediction = tf.reshape(down_project, [self.batch_size, local_max_seq_length, self.output_dim])

            # seq2seq
            encoder_cells = [tf.contrib.rnn.GRUCell(num_units=self.hidden_state_size) for _ in range(self.num_layers)]

            # we stack the cells together and create one big RNN cell
            encoder_cell = tf.contrib.rnn.MultiRNNCell(encoder_cells)

            # Build RNN cell
            # encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_state_size)

            # Run Dynamic RNN
            #   encoder_outputs: [max_time, batch_size, num_units]
            #   encoder_state: [batch_size, num_units]
            self.initial_state = encoder_cell.zero_state(self.batch_size, dtype=tf.float32)

            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                encoder_cell, self.encoder_input_, initial_state=self.initial_state,
                sequence_length=self.encoder_seq_lengths, time_major=False)

            # Build RNN cell
            decoder_cells = [tf.contrib.rnn.GRUCell(num_units=self.hidden_state_size) for _ in range(self.num_layers)]

            # we stack the cells together and create one big RNN cell
            decoder_cell = tf.contrib.rnn.MultiRNNCell(encoder_cells)

            # Helper
            helper = tf.contrib.seq2seq.TrainingHelper(
                self.decoder_input_, self.decoder_seq_lengths, time_major=False)
            # Decoder
            decoder = tf.contrib.seq2seq.BasicDecoder(
                decoder_cell, helper, encoder_state)

            # Dynamic decoding
            decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, ...)
            logits = decoder_outputs.rnn_output

            print(logits.shape)

            outputs_flat = tf.reshape(logits, [-1, self.hidden_state_size])
            down_project = tf.layers.dense(outputs_flat, units=self.output_dim)
            local_max_seq_length = tf.shape(self.input_)[1]
            self.prediction = tf.reshape(down_project, [self.batch_size, local_max_seq_length, self.output_dim])


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

                expanded_mask = tf.expand_dims(self.mask, axis=-1)
                self.loss = tf.losses.mean_squared_error(labels=self.target, predictions=self.prediction, weights=expanded_mask)
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
        # print('Mask shape: ', batch.mask.shape)
        print('SEQ LENGTHS: ', batch.seq_lengths)
        feed_dict = {self.input_: input_padded,
                     self.target: target_padded,
                     self.seq_lengths: batch.seq_lengths,
                     self.mask: batch.mask}

        return feed_dict
