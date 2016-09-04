import argparse
import html
import json
import logging
import random

import sys
from colorbot import data, encoder, decoder, constants

import tensorflow as tf
import numpy as np
from colorbot.data import rgb_to_hex

logger = logging.getLogger(__name__)


def sample():
    args = argparse.ArgumentParser("Train the model.")
    args.add_argument("-n", "--hidden-size", type=int, default=150,
                      help="hidden layer size")
    args.add_argument("-b", "--batch-size", type=int, default=50,
                      help="batch size")
    args.add_argument("data_dir", type=str, help="data directory")

    args = args.parse_args()

    with open("%s/vocab.json" % args.data_dir) as f:
        vocab = data.load_vocab(f)

    session = tf.Session()

    encoder_model = encoder.Encoder(args.hidden_size, len(vocab) // 2)
    decoder_model = decoder.Decoder(args.hidden_size, len(vocab) // 2)

    saver = tf.train.Saver()
    session.run(tf.initialize_all_variables())
    saver.restore(session, "%s/params" % args.data_dir)

    sys.stderr.write("Enter color name: ")
    color_name = input()

    name = "%s%s%s" % (
        constants.START_SYMBOL,
        color_name.strip().lower(),
        constants.END_SYMBOL,
    )

    steps = []

    state = None

    for i in range(len(name)):
        char = name[i]
        enc_char = vocab[char]

        if state is None:
            output, state = session.run(
                [encoder_model.output, encoder_model.final_state],
                feed_dict={
                    encoder_model.input: [[enc_char]],
                    encoder_model.length: [1],
                },
            )
        else:
            output, state = session.run(
                [encoder_model.output, encoder_model.final_state],
                feed_dict={
                    encoder_model.initial_state: state,
                    encoder_model.input: [[enc_char]],
                    encoder_model.length: [1],
                },
            )

        hex_str = rgb_to_hex(*output[0].tolist())

        steps.append((char, hex_str))

    letter_html = ""
    color_html = ""

    for char, hex_str in steps:

        if char == constants.START_SYMBOL:
            char = html.escape("<S>")
        elif char == constants.END_SYMBOL:
            char = html.escape("<E>")

        letter_html += """
        <td>%(char)s</td>
        """ % {"char": char}

        color_html += """
        <td><div class="box" style="background: #%(color)s"></div></td>
        """ % {"color": hex_str}

    page = """<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="UTF-8"/>
        <title>Color Progress</title>
        <style>
            td {
                padding: 2px;
                text-align: center;
                vertical-align: middle;
            }

            .box {
                width: 75px;
                height: 75px;
            }
        </style>
    </head>
    <body>
        <table>
            <tr>%(letters)s</tr>
            <tr>%(colors)s</tr>
        </table>
    </body>
</html>
"""

    print(page % {"letters": letter_html, "colors": color_html})

    exit(0)
