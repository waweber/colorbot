import io
import logging

import numpy as np
import tensorflow as tf
import tweepy
from colorbot import constants
from colorbot.data import rgb_to_hex
from colorbot.decoder import Decoder
from colorbot.encoder import Encoder
from colorbot.twitter.drawing import create_png

logger = logging.getLogger(__name__)


def post_color(name_set, api, vocab, hidden_size, param_path):
    random_color = np.random.rand(3) * 2 - 1

    hex_str = rgb_to_hex(*random_color.tolist())

    logger.info("Naming color %s" % hex_str)

    session = tf.Session()
    encoder = Encoder(hidden_size, len(vocab) // 2)
    model = Decoder(hidden_size, len(vocab) // 2)
    session.run(tf.initialize_all_variables())

    saver = tf.train.Saver()
    saver.restore(session, param_path)

    name = None

    while name is None:
        name_seq = [vocab[constants.START_SYMBOL]]
        state = None

        while (len(name_seq) < 50 and
                       vocab[name_seq[-1]] != constants.END_SYMBOL):
            if state is None:
                state, output = session.run(
                    [model.final_state, model.output],
                    feed_dict={
                        model.state: [random_color.tolist()],
                        model.input: [[name_seq[-1]]],
                        model.length: [1],
                        model.mask: [[1.0]],
                    },
                )
            else:
                state, output = session.run(
                    [model.final_state, model.output],
                    feed_dict={
                        model.initial_state: state,
                        model.input: [[name_seq[-1]]],
                        model.length: [1],
                        model.mask: [[1.0]],
                    },
                )

            val = np.random.rand()

            for i, prob in enumerate(output[0]):
                if prob > val:
                    name_seq.append(i)
                    break
                else:
                    val -= prob

        name_seq_str = "".join(vocab[i] for i in name_seq)

        if (vocab[name_seq[-1]] == constants.END_SYMBOL and
                    name_seq_str not in name_set):
            name = name_seq_str[1:-1]

    session.close()
    tf.reset_default_graph()

    logger.info("Posting name \"%s\"" % name)

    png_data = create_png(*random_color.tolist())

    png_file = io.BytesIO(png_data)

    status = "%s - %s" % (name, hex_str)

    try:
        api.update_with_media("%s.png" % name, status, file=png_file)
    except tweepy.TweepError as e:
        logger.error("Failed to send tweet: %s" % e)
