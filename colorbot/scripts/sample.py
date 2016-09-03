import argparse
import json
import logging
import random

from colorbot import data, encoder, decoder, constants

import tensorflow as tf
import numpy as np

logger = logging.getLogger(__name__)

page_tpl = """<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="UTF-8"/>
        <title>Colors</title>
        <style>
            .box {
                width: 75px;
                height: 75px;
                border: #333 solid 1px;
            }

            td {
                padding: 2px;
            }

            td.name {
                width: 200px;
            }
        </style>
    </head>
    <body>
        <table>
            %(colors)s
        </table>
    </body>
</html>
"""

color_row_tpl = """
<tr>
    %(content)s
</tr>
"""

color_tpl = """
<td><div class="box" style="background: #%(hex)s"/></td>
<td class="name">%(name)s</td>
<td><div class="box" style="background: #%(hex2)s"/></td>
"""


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

    with open("%s/colors.json" % args.data_dir) as f:
        colors_list = json.loads(f.read())

        name_set = {c[0] for c in colors_list}

    session = tf.Session()

    encoder_model = encoder.Encoder(args.hidden_size, len(vocab) // 2)
    decoder_model = decoder.Decoder(args.hidden_size, len(vocab) // 2)

    saver = tf.train.Saver()
    session.run(tf.initialize_all_variables())
    saver.restore(session, "%s/params" % args.data_dir)

    colors = []

    color = (
        2 * random.random() - 1.0,
        2 * random.random() - 1.0,
        2 * random.random() - 1.0,
    )

    while len(colors) < 30:

        # Name color
        name = [vocab[constants.START_SYMBOL]]
        state = None

        while len(name) < 50 and vocab[name[-1]] != constants.END_SYMBOL:
            if state is None:
                output, state = session.run(
                    [decoder_model.output, decoder_model.final_state],
                    feed_dict={
                        decoder_model.state: [color],
                        decoder_model.input: [[name[-1]]],
                        decoder_model.length: [1],
                        decoder_model.mask: [[1.0]],
                    },
                )
            else:
                output, state = session.run(
                    [decoder_model.output, decoder_model.final_state],
                    feed_dict={
                        decoder_model.initial_state: state,
                        decoder_model.input: [[name[-1]]],
                        decoder_model.length: [1],
                        decoder_model.mask: [[1.0]],
                    },
                )

            val = random.random()

            for i in range(output[0].shape[0]):
                if val < output[0][i]:
                    name.append(i)
                    break
                else:
                    val -= output[0][i]

        str_name = "".join(vocab[i] for i in name)

        if str_name[-1] == constants.END_SYMBOL and str_name not in name_set:

            output = session.run(
                encoder_model.output,
                feed_dict={
                    encoder_model.input: [name],
                    encoder_model.length: [len(name)],
                },
            )

            hex = data.rgb_to_hex(*output[0].tolist())

            colors.append((data.rgb_to_hex(*color), str_name[1:-1], hex))

            color = (
                2 * random.random() - 1.0,
                2 * random.random() - 1.0,
                2 * random.random() - 1.0,
            )

    content = ""

    for i in range(len(colors) // 3):

        row = ""

        for j in range(3):
            in_hex, name, out_hex = colors[i * 3 + j]

            color_str = color_tpl % {
                "hex": in_hex,
                "name": name,
                "hex2": out_hex,
            }

            row += color_str

        row_str = color_row_tpl % {
            "content": row,
        }

        content += row_str

    final = page_tpl % {
        "colors": content,
    }

    print(final)
    exit(0)
