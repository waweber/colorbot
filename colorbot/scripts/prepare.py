import argparse
import json
import logging
import os
from colorbot import data, scrape

logger = logging.getLogger(__name__)


def prepare_data():
    args = argparse.ArgumentParser("Prepare the training dataset")
    args.add_argument("data_dir", type=str, help="data directory")

    args = args.parse_args()

    if not os.path.exists(args.data_dir):
        logger.debug("Creating data dir: %s" % args.data_dir)
        os.mkdir(args.data_dir)

    colors = []

    for source in scrape.color_sources:
        logger.info("Getting colors from source \"%s\"", source.__name__)

        count = 0

        for color in source():
            colors.append(color)
            count += 1

        logger.info("Got %d colors" % count)

    def sorter(x):
        mag = x.r ** 2 * x.g ** 2 * x.b ** 2
        return mag, x.r, x.g, x.b

    colors.sort(key=sorter)

    vocab = data.build_vocab(colors)

    logger.info("Saving vocab to: %s/vocab.json" % args.data_dir)
    with open("%s/vocab.json" % args.data_dir) as f:
        data.save_vocab(vocab, f)

    logger.info("Saving colors to %s/colors.json" % args.data_dir)
    with open("%s/colors.json" % args.data_dir) as f:
        f.write(json.dumps(colors))

    logger.info("Saved %d colors" % len(colors))

    exit(0)
