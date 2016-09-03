import logging
import re

import tweepy
from colorbot import constants
from colorbot.data import hex_to_rgb
from colorbot.decoder import Decoder
from colorbot.encoder import Encoder

import tensorflow as tf
import numpy as np

logger = logging.getLogger(__name__)


def name_color(hex, status_id, screen_name, api, name_set, vocab, hidden_size,
               param_path):
    logger.info("Naming color %s" % hex)

    r, g, b = hex_to_rgb(hex)

    session = tf.Session()

    encoder = Encoder(hidden_size, len(vocab) // 2)
    decoder = Decoder(hidden_size, len(vocab) // 2)

    session.run(tf.initialize_all_variables())

    saver = tf.train.Saver()
    saver.restore(session, param_path)

    name = None
    tries = 50

    while name is None and tries > 0:
        tries -= 1

        name_seq = [vocab[constants.START_SYMBOL]]
        state = None

        while (len(name_seq) < 50 and
                       vocab[name_seq[-1]] != constants.END_SYMBOL):
            if state is None:
                state, output = session.run(
                    [decoder.final_state, decoder.output],
                    feed_dict={
                        decoder.state: [[r, g, b]],
                        decoder.input: [[name_seq[-1]]],
                        decoder.length: [1],
                        decoder.mask: [[1.0]],
                    },
                )
            else:
                state, output = session.run(
                    [decoder.final_state, decoder.output],
                    feed_dict={
                        decoder.initial_state: state,
                        decoder.input: [[name_seq[-1]]],
                        decoder.length: [1],
                        decoder.mask: [[1.0]],
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

    if name is None:
        logger.warn("Giving up on naming")
    else:
        logger.info("Replying with name %s" % name)
        status = "@%s %s" % (screen_name, name)

        try:
            api.update_status(status, status_id)
        except tweepy.TweepError as e:
            logger.warn("Failed replying with color name: %s" % e)


class StreamListener(tweepy.StreamListener):
    def __init__(self, api, name_set, vocab, hidden_size, param_path):
        self.api = api
        self.name_set = name_set
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.param_path = param_path
        self._my_id = None

    @property
    def my_id(self):
        if self._my_id is None:
            me = self.api.me()
            self._my_id = me.id_str

        return self._my_id

    def on_error(self, status_code):
        logger.error("Stream received error code %d" % status_code)
        return False

    def on_status(self, status):
        id = status.id_str
        author_id = status.user.id_str
        screen_name = status.user.screen_name
        msg = status.text

        if author_id == self.my_id:
            # Ignore, this is our own tweet
            pass
        else:
            # Find a color hex code?
            hex_match = re.search("^@[a-zA-Z0-9_]+(\s+|\s*#)([a-fA-F0-9]{6})",
                                  msg)

            # Find a name?
            name_match = re.search("^@[a-zA-Z0-9_]+\s+(.*)", msg)

            if hex_match is not None:
                hex_str = hex_match.group(2)
            elif name_match is not None:
                name = hex_match.group(1).strip().lower()

                # Remove unknown characters
                filtered_name = "".join(c for c in name if c in self.vocab)

                input_name = (constants.START_SYMBOL + filtered_name +
                              constants.END_SYMBOL)


def run(name_set, api, vocab, hidden_size, param_path):
    pass
