import argparse

import tweepy
from colorbot.twitter.auth import save_auth


def auth():
    args = argparse.ArgumentParser(description="Get Twitter auth credentials")
    args.add_argument("output_file", type=str,
                      help="where to save the credentials")

    args = args.parse_args()

    consumer_key = input("Consumer Key: ")
    consumer_secret = input("Consumer Secret: ")

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret,
                               callback="http://127.0.0.1/")
    url = auth.get_authorization_url()

    print("Authorize at: %s" % url)

    verifier = input("Verifier: ")

    try:
        auth.get_access_token(verifier)
    except tweepy.TweepError:
        print("Authorization failed")
        exit(2)

    with open(args.output_file, "w") as f:
        save_auth(auth, f)

    exit(0)
