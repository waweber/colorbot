import tensorflow as tf
from colorbot import tf_util, constants


class Encoder(object):
    def __init__(self, hidden_size, vocab_size):
        cell = tf.nn.rnn_cell.GRUCell(hidden_size)

        embed_params = tf_util.weights([vocab_size, hidden_size])

        state_w = tf_util.weights([constants.COLOR_SIZE, hidden_size])
        state_b = tf_util.bias([hidden_size])

        output_w = tf_util.weights([hidden_size, vocab_size])
        output_b = tf_util.bias([vocab_size])

        # inputs
        state = tf.placeholder(tf.float32, [None, constants.COLOR_SIZE])
        input = tf.placeholder(tf.int32, [None, None])
        length = tf.placeholder(tf.int32, [None])
        mask = tf.placeholder(tf.float32, [None, None])
        label = tf.placeholder(tf.int32, [None, None])

        # Project the state
        initial_state = tf.nn.elu(tf.nn.xw_plus_b(state, state_w, state_b))

        # Lookup inputs
        lookup_input = tf.nn.embedding_lookup(embed_params, input)

        # RNN
        output, final_state = tf.nn.dynamic_rnn(
            cell,
            lookup_input,
            sequence_length=length,
            initial_state=initial_state,
            scope="Decoder",
        )

        # Reshape outputs
        #   Before: [batch size, seq len, hidden size]
        #   After: [batch size * seq len, hidden size]
        reshaped_output = tf.reshape(output, [-1, hidden_size])

        # Project output
        logits = tf.nn.xw_plus_b(reshaped_output, output_w, output_b)

        # Loss/softmax
        output = tf.nn.softmax(logits)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits,
            tf.reshape(label, [-1]),
        )

        masked_loss = tf.mul(
            loss,
            tf.reshape(mask, [-1]),
        )

        trainer = tf.train.AdamOptimizer(0.001).minimize(
            tf.reduce_sum(masked_loss) / tf.reduce_sum(mask),
        )

        self.state = state
        self.input = input
        self.length = length
        self.mask = mask
        self.label = label
        self.output = output
        self.final_state = final_state
        self.loss = masked_loss
        self.trainer = trainer
