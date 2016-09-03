import json
import logging

import tweepy

logger = logging.getLogger(__name__)


def get_auth(f):
    """Load auth credentials from a json file.

    Args:
        f: The file object

    Returns:
        A tweepy auth object, or None if loading fails.
    """
    try:
        data = f.read()
        json_data = json.loads(data)

        consumer_key = json_data["consumer_key"]
        consumer_secret = json_data["consumer_secret"]
        access_token = json_data["access_token"]
        access_token_secret = json_data["access_token_secret"]

    except (json.JSONDecodeError, KeyError):
        logger.error("Loading Twitter credentials failed")
        return None

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    return auth


def save_auth(auth, f):
    """Save auth credentials to a json file.

    Args:
        auth: The OAuthHandler
        f: The file object to write to
    """

    data = {
        "consumer_key": auth.consumer_key,
        "consumer_secret": auth.consumer_secret,
        "access_token": auth.access_token,
        "access_token_secret": auth.access_token_secret,
    }

    data_str = json.dumps(data, sort_keys=True, indent=4)
    f.write(data_str)
