import logging
import re

import tweepy
from colorbot import constants

logger = logging.getLogger(__name__)


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
