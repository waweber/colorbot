import argparse
import json
import logging
import random

from colorbot import data, encoder, decoder

import tensorflow as tf
import numpy as np

logger = logging.getLogger(__name__)


def train():
    args = argparse.ArgumentParser("Train the model.")
    args.add_argument("-n", "--hidden-size", type=int, default=150,
                      help="hidden layer size")
    args.add_argument("-b", "--batch-size", type=int, default=50,
                      help="batch size")
    args.add_argument("data_dir", type=str, help="data directory")

    args = args.parse_args()

    logger.info("Loading vocab")
    with open("%s/vocab.json" % args.data_dir) as f:
        vocab = data.load_vocab(f)

    logger.info("Loading colors")
    colors = []
    with open("%s/colors.json" % args.data_dir) as f:
        json_list = json.loads(f.read())
        for color_tuple in json_list:
            colors.append(data.Color(*color_tuple))

    random.shuffle(colors)

    batches = data.yield_batches(colors, args.batch_size)
    prepared_batches = list(data.prepare_batch(b, vocab) for b in batches)

    logger.info("Building model")
    session = tf.Session()
    encoder_model = encoder.Encoder(args.hidden_size, len(vocab) // 2)
    decoder_model = decoder.Decoder(args.hidden_size, len(vocab) // 2)

    session.run(tf.initialize_all_variables())

    saver = tf.train.Saver()

    try:
        logger.warn("Starting training, stop with Ctrl-C")

        epoch = 0

        while True:
            epoch += 1

            enc_loss_num = 0
            enc_loss_denom = 0
            dec_loss_num = 0
            dec_loss_denom = 0

            for batch in prepared_batches:
                enc_loss, _ = session.run(
                    [encoder_model.loss, encoder_model.trainer],
                    feed_dict={
                        encoder_model.input: batch.encoder_input,
                        encoder_model.length: batch.encoder_length,
                        encoder_model.target: batch.encoder_target,
                    },
                )

                enc_loss_num += np.sum(enc_loss)
                enc_loss_denom += enc_loss.shape[0]

                dec_loss, _ = session.run(
                    [decoder_model.loss, decoder_model.trainer],
                    feed_dict={
                        decoder_model.state: batch.decoder_state,
                        decoder_model.input: batch.decoder_input,
                        decoder_model.length: batch.decoder_length,
                        decoder_model.mask: batch.decoder_mask,
                        decoder_model.label: batch.decoder_label,
                    },
                )

                dec_loss_num += np.sum(dec_loss)
                dec_loss_denom += dec_loss.shape[0]

            logger.info("Epoch %d, encoder loss %.3f, decoder loss %.3f" % (
                epoch, enc_loss_num / enc_loss_denom,
                dec_loss_num / dec_loss_denom))

    except KeyboardInterrupt:
        logger.warn("Stopping training")

    saver.save(session, "%s/params" % args.data_dir)

    exit(0)
