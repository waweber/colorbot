import tensorflow as tf
from colorbot import tf_util, constants


class Encoder(object):
    def __init__(self, hidden_size, vocab_size):
        cell = tf.nn.rnn_cell.GRUCell(hidden_size)
        state_size = cell.state_size

        output_w = tf_util.weights([state_size, constants.COLOR_SIZE])
        output_b = tf_util.bias([constants.COLOR_SIZE])

        embed_params = tf_util.weights([vocab_size, hidden_size])

        # Placeholders
        input = tf.placeholder(tf.int32, [None, None])
        length = tf.placeholder(tf.int32, [None])
        target = tf.placeholder(tf.float32, [None, constants.COLOR_SIZE])

        # lookup input
        lookup_input = tf.nn.embedding_lookup(embed_params, input)

        # RNN
        _, final_state = tf.nn.dynamic_rnn(
            cell,
            lookup_input,
            sequence_length=length,
            dtype=tf.float32,
            scope="Encoder",
        )

        # Project final state
        proj = tf.nn.tanh(tf.nn.xw_plus_b(final_state, output_w, output_b))

        # Loss (MSE)
        sse = tf.reduce_sum(tf.square(tf.sub(proj, target)), 1)
        mse = sse / constants.COLOR_SIZE

        trainer = tf.train.AdamOptimizer(0.001).minimize(
            tf.reduce_sum(mse),
        )

        self.input = input
        self.length = length
        self.target = target
        self.output = proj
        self.loss = sse
        self.trainer = trainer
