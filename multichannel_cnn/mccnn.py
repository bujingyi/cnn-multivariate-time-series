import tensorflow as tf
import os


class MCCNN:
    """
    Multi-Channel CNN for multivariante time series classification
    """
    def __init__(
        self,
        x_dim,
        y_dim,
        seqlen=32,
        scope='MCCNN',
        initial_learning_rate=1e-4,
        decay_steps=10000,
        decay_rate=0.9,
        summaries_dir=None, 
    ):
    """
    MCCNN initializer
    :param x_dim: input feature x dimension
    :param y_dim: input feature y dimension
    :param scope: TensorFlow variable scope
    :param initial_learning_rate: initial gradient descent learning rate
    :param decay_steps: GD learning rate decay steps
    :param decay_rate: GD learning rate decay rate
    :param summaries_dir: TensorBoard summaries

    """
    self.x_dim = x_dim
    self.y_dim = y_dim
    self.seqlen = seqlen
    self.initial_learning_rate = initial_learning_rate
    self.decay_steps = decay_steps
    self.decay_rate = decay_rate

    # build tf computation graph and tf summaries
    with tf.variable_scope(self.scope):
        self._build_model()
        if summaries_dir:
            summaries_dir = os.path.join(summaries_dir, 'summaries_{}'.format(scope))
            if not os.path.exists(summaries_dir):
                os.makedirs(summaries_dir)
            self.summary_writer = tf.summary.FileWriter(summaries_dir)


    def _build_model(self):
        """
        Build computation graph
        """
        # placeholders
        self.x_ph = tf.placeholder(tf.float32, [None, None, self.x_dim], name='inputs')
        self.y_ph = tf.placeholder(tf.float32, [None, self.y_dim], name='targets')

        # build 1D convolutional blocks for each channel
        cnn_outputs = []  # a list to collect multi-channel 1D cnn outputs
        for channel in range(self.x_dim):
            with tf.name_scope('Conv_Maxpool_{}'.format(channel)):
                # filters and biases of 1D conv layers
                with tf.variable_scope('conv_maxpool_{}'.format(channel)):
                    filter1 = tf.get_variable('filter1', [5, 1, 8])
                    bias1 = tf.get_variable('bias1', [8], initializer=tf.constant_initializer(0.0))
                    filter2 = tf.get_variable('filter2', [3, 8, 4])
                    bias2 = tf.get_variable('bias2', [4], initializer=tf.constant_initializer(0.0))

                inputs = tf.reshape(self.x_ph[:, :, channel], [-1, self.seqlen, 1])

                # 1D cnn block 1, seqlen: 32 --> 14
                # filter shape [filter_width, in_channels, out_channels]
                conv1 = tf.nn.conv1d(
                    value=inputs, 
                    filters=filter1, 
                    stride=1, 
                    padding='VALID', 
                    name='conv1d_1'
                )
                seqlen = self.seqlen - 4
                h1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1), name='h1_relu')
                h1 = tf.reshape(h1, shape=[-1, 1, self.seqlen - 4, 8])
                avgpool1 = tf.nn.avg_pool(
                    value=h1, 
                    ksize=[1, 1, 2, 1],  
                    strides=[1, 1, 2, 1],
                    padding='VALID', 
                    name='avg_pool_1'
                )
                avgpool1 = tf.reshape(avgpool1, shape=[-1, 14, 8])

                # 1D cnn block 2, seqlen: 14 --> 6
                conv2 = tf.nn.conv1d(
                    value=avgpool1, 
                    filters=filter2W, 
                    stride=1, 
                    padding='VALID', 
                    name='conv1d_2'
                )
                h2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2))
                h2 = tf.reshape(h2, shape = [-1, 1, 12, 4])
                avgpool2 = tf.nn.avg_pool(
                    value=h2, 
                    ksize=[1, 1, 2, 1], 
                    strides=[1, 1, 2, 1], 
                    padding='VALID', 
                    name='avg_pool_2'
                )
                avgpool2 = tf.reshape(avgpool2, shape=[-1, 1, 6, 4])

                # collect multi-channel outputs
                cnn_outputs.append(avgpool2)

        # Combine all channels' cnn outputs
        cnn_outputs = tf.concat(cnn_outputs, axis=3)
        num_filters = self.x_dim * 4
        cnn_outputs_flat = tf.reshape(cnn_outputs, [-1, num_filters_total * 6])  # [batch, x_dim * 24]

        # fully connected layer
        with tf.name_scope('Dense'):
            with tf.variable_scope('logits'):
                dense1_w = tf.get_variable('wd1', [6 * 4 * 16, 16])
                dense1_b = tf.get_variable('bd1', [16], initializer=tf.constant_initializer(0.0))
            fc1 = tf.matmul(cnn_outputs_flat, dense1_w) + dense1_b
            dense_outputs = tf.nn.relu(fc1)
            # dropout
            dense1_dropout = tf.nn.dropout(dense_outputs, dropout, name='dropout')

        # final outputs
        with tf.variable_scope('Logits'):
            logits_w = tf.get_variable('logits_w', [16, self.y_dim])
            logits_b = tf.get_variable('logits_b', [self.y_dim], initializer=tf.constant_initializer(0.0))

        logits = tf.matmul(dense1_dropout, logits_w) + logits_b

        # predictions
        with tf.name_scope('Prediction'):
            self.preds = tf.softmax(logits=logits) 

        # training with gradient descent, global variables
        with tf.name_scope('Global'):
            global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
            self.learning_rate = tf.train.exponential_decay(
                learning_rate=self.initial_learning_rate,
                global_step=global_step,
                decay_steps=self.decay_steps,
                decay_rate=self.decay_rate
            )
        with tf.name_scope('Loss'):
            self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.y_ph))
        with tf.name_scope('Train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss, global_step=global_step)

        # summaries
        self.summaries = tf.summary.merge([
            tf.summary.scalar('loss', self.loss),
        ])


    def update(self, sess, x, y):
        """
        Updates the acceptor towards the given targets
        :param sess: TensorFlow session
        :param x: input x of shape [batch_size, known_length]
        :param y: targets to predict of shape[batch_size, pred_length]
        :return: loss
        """
        feed_dict = {
            self.x_ph: x, 
            self.y_ph: y
        }
        # update
        preds, loss, _, learning_rate, global_step, summaries = sess.run(
            [
                self.preds,
                self.loss, 
                self.train_op, 
                self.learning_rate, 
                tf.train.get_global_step(), 
                self.summaries
            ],
            feed_dict
        )
        if self.summary_writer and save:
            self.summary_writer.add_summary(summaries, global_step)
        return preds, loss

    def predict(self, sess, x, y, init_state):
        """
        Updates the acceptor towards the given targets
        :param sess: TensorFlow session
        :param x: input x of shape [batch_size, known_length]
        :param y: targets to predict of shape[batch_size, pred_length]
        :return: predictions
        """
        feed_dict = {
            self.x_ph: x, 
            self.y_ph: y
        }
        return sess.run(self.preds, feed_dict)