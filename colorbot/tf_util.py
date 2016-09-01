import tensorflow as tf


def weights(shape):
    """Create a weight variable.
    """
    input_size = shape[0]
    var = tf.Variable(tf.truncated_normal(
        shape, stddev=tf.sqrt(2.0 / input_size)
    ))

    return var


def bias(shape):
    """Create a bias variable.
    """
    var = tf.Variable(tf.zeros(shape))
    return var
