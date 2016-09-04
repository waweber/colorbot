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
    """Class to hold global application state.

    Attributes:
        api: The Tweepy API instance
        name_set (set): A set of strings of existing color names
        vocab: The encoder/decoder vocabulary
        hidden_size (int): The model's hidden layer size
        param_path (str): Path to the saved model parameters
        stop (bool): True if the worker threads should stop
        tasks: List of dicts describing tasks to do
        lock: Mutex to control shared access to these attributes
        has_tasks: Condition notified when tasks is changed
    """

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
    """Name a color and post to Twitter.

    Acquire the lock before using.

    Args:
        hex (str): The hex color code string (6 chars)
        status_id (str): The ID of the status to reply to
        screen_name (str): The screen name to reply to
        global_state: A global state instance
    """
    logger.info("Naming color %s" % hex)

    r, g, b = hex_to_rgb(hex)

    # Set up session and models
    session = tf.Session()

    encoder = Encoder(global_state.hidden_size,
                      len(global_state.vocab) // 2)
    decoder = Decoder(global_state.hidden_size,
                      len(global_state.vocab) // 2)

    session.run(tf.initialize_all_variables())

    # Load params from file
    saver = tf.train.Saver()
    saver.restore(session, global_state.param_path)

    # Name color
    name = None
    tries = 50

    # Try up to `tries` times to get a valid name
    while name is None and tries > 0:
        tries -= 1

        name_seq = [global_state.vocab[constants.START_SYMBOL]]
        state = None

        # Sample from model until:
        #   - We get a sequence longer than 50 (too long, quit), OR
        #   - The returned symbol is the end symbol
        while (len(name_seq) < 50 and
                       global_state.vocab[
                           name_seq[-1]] != constants.END_SYMBOL):

            if state is None:
                # Provide color to generate first hidden state
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
                # Use the last hidden state
                state, output = session.run(
                    [decoder.final_state, decoder.output],
                    feed_dict={
                        decoder.initial_state: state,
                        decoder.input: [[name_seq[-1]]],
                        decoder.length: [1],
                        decoder.mask: [[1.0]],
                    },
                )

            # Weighted random pick from output PMF
            val = np.random.rand()

            for i, prob in enumerate(output[0]):
                if prob > val:
                    name_seq.append(i)
                    break
                else:
                    val -= prob

        # Turn the output sequence into strings
        name_seq_str = "".join(global_state.vocab[i] for i in name_seq)

        # If name generation is successful, set name
        # We skip this and try again if the loop exited before generating the
        # end symbol, or if it generated an existing color name
        if (global_state.vocab[name_seq[-1]] == constants.END_SYMBOL and
                    name_seq_str not in global_state.name_set):
            name = name_seq_str[1:-1]

    # Clean up session
    session.close()
    tf.reset_default_graph()

    # Try to tweet
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
    """Guess a color value, upload it to twitter.

    Acquire the lock before using.

    Args:
        name (str): The name of the color
        status_id (str): The ID of the status to reply to
        screen_name (str): The screen name to reply to
        global_state: A global state instance
    """

    logger.info("Guessing color for \"%s\"" % name)

    # Set up model
    session = tf.Session()

    encoder = Encoder(global_state.hidden_size,
                      len(global_state.vocab) // 2)
    decoder = Decoder(global_state.hidden_size,
                      len(global_state.vocab) // 2)

    session.run(tf.initialize_all_variables())

    # Load model parameters
    saver = tf.train.Saver()
    saver.restore(session, global_state.param_path)

    # Feed name into model
    enc_name = [global_state.vocab[c] for c in name]

    output = session.run(
        encoder.output,
        feed_dict={
            encoder.input: [enc_name],
            encoder.length: [len(enc_name)],
        },
    )

    # Turn model output into hex
    r, g, b = output[0].tolist()
    hex_str = rgb_to_hex(r, g, b)

    # Clean up model
    session.close()
    tf.reset_default_graph()

    logger.info("Guessed %s" % hex_str)

    # Create PNG file
    png_data = create_png(r, g, b)
    png_file = io.BytesIO(png_data)

    # Upload to twitter
    status = "@%s %s - %s" % (screen_name, name[:50], hex_str)

    try:
        global_state.api.update_with_media("%s.png" % name, status,
                                           in_reply_to_status_id=status_id,
                                           file=png_file)
    except tweepy.TweepError as e:
        logging.error("Failed to tweet guessed color: %s" % e)


class StreamListener(tweepy.StreamListener):
    def __init__(self, state):
        self.state = state
        self.api = self.state.api
        self._my_id = None

    @property
    def my_id(self):
        if self._my_id is None:
            with self.state.lock:
                logger.debug("Looking up own ID")
                me = self.state.api.me()
                logger.debug("We are %s (%s)" % (me.id_str, me.screen_name))
            self._my_id = me.id_str

        return self._my_id

    def on_error(self, status_code):
        logger.error("Stream received error code %d" % status_code)
        return False

    def on_disconnect(self, notice):
        return False

    def on_status(self, status):
        id = status.id_str
        author_id = status.user.id_str
        screen_name = status.user.screen_name
        msg = status.text

        logger.debug("Got status: %s" % msg)

        if author_id == self.my_id:
            # Ignore, this is our own tweet
            pass
        else:
            # Find a color hex code?
            hex_match = re.search(
                "^@[a-zA-Z0-9_]+(\s+|\s*#)([a-fA-F0-9]{6})\s*$", msg)

            # Find a name?
            name_match = re.search("^@[a-zA-Z0-9_]+\s+(.+)", msg)

            if hex_match is not None:
                hex_str = hex_match.group(2)

                # Tell the worker thread to handle this
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
                name = name_match.group(1).strip().lower()

                # Remove unknown characters
                filtered_name = "".join(
                    c for c in name if c in self.state.vocab)

                # Add start/end symbols
                input_name = (constants.START_SYMBOL + filtered_name +
                              constants.END_SYMBOL)

                # Tell worker thread to guess
                task = {
                    "type": "guess",
                    "name": input_name,
                    "status_id": id,
                    "screen_name": screen_name,
                }

                with self.state.has_tasks:
                    self.state.tasks.append(task)
                    self.state.has_tasks.notify()


def worker(state):
    """Twitter reply worker thread.

    Args:
        state: The global state instance
    """

    while True:
        with state.has_tasks:
            # Wait for state.has_tasks to be notified
            while state.stop is False and len(state.tasks) == 0:
                state.has_tasks.wait()

            # Check if stop is requested
            if state.stop is not False:
                logger.debug("Worker thread exiting")
                return  # Stop

            # get task
            task = state.tasks.pop()

            logger.debug("Worker thread handling task: %r" % task)

            # Handle task
            if task["type"] == "name":
                name_color(task["hex"], task["status_id"], task["screen_name"],
                           state)
            elif task["type"] == "guess":
                guess_color(task["name"], task["status_id"],
                            task["screen_name"], state)

        # Don't try to handle anything for a short while
        time.sleep(10)


def run(auth, name_set, api, vocab, hidden_size, param_path):
    """Run the bot.

    Args:
        auth: Tweepy auth handler instance
        name_set: Set of real color names
        api: Tweepy API instance
        vocab: The model vocabulary dict
        hidden_size (int): The model's hidden size
        param_path (str): Path of the model parameters file
    """

    state = GlobalState()
    state.api = api
    state.name_set = name_set
    state.vocab = vocab
    state.hidden_size = hidden_size
    state.param_path = param_path

    def term_handler(*args):
        raise KeyboardInterrupt("SIGTERM")

    signal.signal(signal.SIGTERM, term_handler)

    listener = StreamListener(state)
    stream = tweepy.Stream(auth, listener)

    logger.info("Starting worker thread")
    worker_thread = threading.Thread(target=worker, args=(state,))
    worker_thread.start()

    logger.info("Starting twitter stream")

    try:
        stream.userstream()
    except KeyboardInterrupt:
        pass

    stream.disconnect()

    with state.has_tasks:
        state.stop = True
        state.has_tasks.notify_all()

    worker_thread.join()

    logger.info("Exiting")
    exit(0)
