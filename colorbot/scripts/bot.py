import argparse
import json

import tweepy
from colorbot.data import load_vocab
from colorbot.twitter.auth import get_auth
from colorbot.twitter.bot import run


def bot():
    args = argparse.ArgumentParser(description="Run the bot")
    args.add_argument("-n", "--hidden-size", type=int, default=150,
                      help="number of hidden units")
    args.add_argument("credential_file", type=str,
                      help="path to file containing credentials")
    args.add_argument("data_dir", type=str, help="path to data directory")

    args = args.parse_args()

    with open("%s/vocab.json" % args.data_dir) as f:
        vocab = load_vocab(f)

    with open("%s/colors.json" % args.data_dir) as f:
        color_list = json.loads(f.read())

        name_set = {c[0] for c in color_list}

    with open(args.credential_file) as f:
        auth = get_auth(f)

    api = tweepy.API(auth_handler=auth)

    run(auth, name_set, api, vocab, args.hidden_size,
        "%s/params" % args.data_dir)
