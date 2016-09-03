import datetime
import io
import logging
import re
import signal
import threading

import time
import tweepy
from colorbot import constants
from colorbot.data import hex_to_rgb, rgb_to_hex
from colorbot.decoder import Decoder
from colorbot.encoder import Encoder

import tensorflow as tf
import numpy as np
from colorbot.twitter.drawing import create_png

logger = logging.getLogger(__name__)


class GlobalState(object):
    def __init__(self):
        self.api = None
        self.name_set = None
        self.vocab = None
        self.hidden_size = None
        self.param_path = None

        self.stop = False

        self.tasks = []

        self.lock = threading.Lock()
        self.has_tasks = threading.Condition(lock=self.lock)


def name_color(hex, status_id, screen_name, global_state):
    with global_state.lock:
        logger.info("Naming color %s" % hex)

        r, g, b = hex_to_rgb(hex)

        session = tf.Session()

        encoder = Encoder(global_state.hidden_size,
                          len(global_state.vocab) // 2)
        decoder = Decoder(global_state.hidden_size,
                          len(global_state.vocab) // 2)

        session.run(tf.initialize_all_variables())

        saver = tf.train.Saver()
        saver.restore(session, global_state.param_path)

        name = None
        tries = 50

        while name is None and tries > 0:
            tries -= 1

            name_seq = [global_state.vocab[constants.START_SYMBOL]]
            state = None

            while (len(name_seq) < 50 and
                           global_state.vocab[
                               name_seq[-1]] != constants.END_SYMBOL):
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

            name_seq_str = "".join(global_state.vocab[i] for i in name_seq)

            if (global_state.vocab[name_seq[-1]] == constants.END_SYMBOL and
                        name_seq_str not in global_state.name_set):
                name = name_seq_str[1:-1]

        session.close()
        tf.reset_default_graph()

        if name is None:
            logger.warn("Giving up on naming")
        else:
            logger.info("Replying with name %s" % name)
            status = "@%s %s" % (screen_name, name)

            try:
                global_state.api.update_status(status, status_id)
            except tweepy.TweepError as e:
                logger.error("Failed replying with color name: %s" % e)


def guess_color(name, status_id, screen_name, global_state):
    with global_state.lock:
        logger.info("Guessing color for \"%s\"" % name)

        session = tf.Session()

        encoder = Encoder(global_state.hidden_size,
                          len(global_state.vocab) // 2)
        decoder = Decoder(global_state.hidden_size,
                          len(global_state.vocab) // 2)

        session.run(tf.initialize_all_variables())

        saver = tf.train.Saver()
        saver.restore(session, global_state.param_path)

        enc_name = [global_state.vocab[c] for c in name]

        output = session.run(
            encoder.output,
            feed_dict={
                encoder.input: [enc_name],
                encoder.length: [len(enc_name)],
            },
        )

        r, g, b = output[0].tolist()
        hex_str = rgb_to_hex(r, g, b)

        session.close()
        tf.reset_default_graph()

        logger.info("Guessed %s" % hex_str)

        png_data = create_png(r, g, b)
        png_file = io.BytesIO(png_data)

        status = "@%s %s - %s" % (screen_name, name[:50], hex_str)

        try:
            global_state.api.update_status_with_media("%s.png" % name, status,
                                                      status_id,
                                                      file=png_file)
        except tweepy.TweepError as e:
            logging.error("Failed to tweet guessed color: %s" % e)


class StreamListener(tweepy.StreamListener):
    def __init__(self, state):
        self.state = state
        self._my_id = None

    @property
    def my_id(self):
        if self._my_id is None:
            with self.state.lock:
                me = self.state.api.me()
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
                task = {
                    "type": "name",
                    "hex": hex_str,
                    "status_id": id,
                    "screen_name": screen_name,
                }

                with self.state.has_tasks:
                    self.state.tasks.append(task)
                    self.state.has_tasks.notify()

            elif name_match is not None:
                name = hex_match.group(1).strip().lower()

                # Remove unknown characters
                filtered_name = "".join(
                    c for c in name if c in self.state.vocab)

                input_name = (constants.START_SYMBOL + filtered_name +
                              constants.END_SYMBOL)

                task = {
                    "type": "name",
                    "name": input_name,
                    "status_id": id,
                    "screen_name": screen_name,
                }

                with self.state.has_tasks:
                    self.state.tasks.append(task)
                    self.state.has_tasks.notify()


def worker(state):
    while True:
        with state.has_tasks:
            while state.stop is False and len(state.tasks) == 0:
                state.has_tasks.wait()

            if state.stop is not False:
                return  # Stop

            # get task
            task = state.tasks.pop()

            if task["type"] == "name":
                name_color(task["hex"], task["status_id"], task["screen_name"],
                           state)
            elif task["type"] == "guess":
                guess_color(task["name"], task["status_id"],
                            task["screen_name"], state)


def run(auth, name_set, api, vocab, hidden_size, param_path):
    state = GlobalState()
    state.api = api
    state.name_set = name_set
    state.vocab = vocab
    state.hidden_size = hidden_size
    state.param_path = param_path

    listener = StreamListener(state)
    stream = tweepy.Stream(auth, listener)

    def int_handler(*args):
        with state.has_tasks:
            state.stop = True
            state.has_tasks.notify_all()

        stream.disconnect()

    signal.signal(signal.SIGINT, int_handler)
    signal.signal(signal.SIGTERM, int_handler)

    worker_thread = threading.Thread(target=worker, args=(state,))
    worker_thread.run()

    stream.userstream(async=True)

    worker_thread.join()
