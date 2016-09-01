"""Module related to color data and samples.
"""
import json
from collections import namedtuple
import numpy as np

from colorbot import constants

Color = namedtuple("Color", ("name", "r", "g", "b"))

Batch = namedtuple("Batch", (
    "encoder_input",
    "encoder_length",
    "encoder_target",
    "decoder_state",
    "decoder_input",
    "decoder_length",
    "decoder_mask",
    "decoder_label",
))


def build_vocab(colors):
    """Build a vocabulary from given colors.

    Args:
        colors: Iterable of Colors

    Returns:
        A dict mapping char indices to characters and vice versa.
    """
    char_set = set()

    for color in colors:
        char_set.update(color.name)

    char_list = sorted(char_set)

    vocab = {i: c for i, c in enumerate(char_list)}
    vocab.update({c: i for i, c in enumerate(char_list)})

    return vocab


def save_vocab(vocab, f):
    """Save the vocab to a file.
    """
    f.write(json.dumps(vocab, sort_keys=True, indent=2))


def load_vocab(f):
    """Load the vocab from a file.
    """
    return json.loads(f.read())


def hex_to_rgb(txt):
    """Turn a hex color code string into an RGB tuple.

    RGB tuples are floats on the interval [-1.0, 1.0].

    Args:
        txt (str): The hex string (6 chars, no #)

    Returns:
        A 3-tuple of R, G, and B floats.
    """
    r_txt = txt[0:2]
    g_txt = txt[2:4]
    b_txt = txt[4:6]

    r_int = int(r_txt, base=16)
    g_int = int(g_txt, base=16)
    b_int = int(b_txt, base=16)

    r = 2 * r_int / 255 - 1.0
    g = 2 * g_int / 255 - 1.0
    b = 2 * b_int / 255 - 1.0

    return r, g, b


def rgb_to_hex(r, g, b):
    """Turn an RGB float tuple into a hex code.

    Args:
        r (float): R value
        g (float): G value
        b (float): B value

    Returns:
        str: A hex code (no #)
    """
    r_int = round((r + 1.0) / 2 * 255)
    g_int = round((g + 1.0) / 2 * 255)
    b_int = round((b + 1.0) / 2 * 255)

    r_txt = "%02x" % r_int
    b_txt = "%02x" % b_int
    g_txt = "%02x" % g_int

    return r_txt + b_txt + g_txt


def yield_batches(iter, batch_size):
    """Yield items from iter grouped into lists.

    Args:
        iter: The iterable
        batch_size (int): Yield lists no larger than this
    """
    batch = []

    for item in iter:
        batch.append(item)

        if len(batch) == batch_size:
            yield batch
            batch = []

    if len(batch) > 0:
        yield batch


def prepare_batch(batch, vocab):
    """Turn a list of colors into a Batch object.

    Args:
        batch: The list of Color objects
        vocab: The vocabulary dict

    Returns:
        A Batch object.
    """
    batch_size = len(batch)
    max_len = max(len(c.name) for c in batch)

    encoder_input = np.zeros([batch_size, max_len], np.int32)
    encoder_length = np.zeros([batch_size], np.int32)
    encoder_target = np.zeros([batch_size, constants.COLOR_SIZE], np.float32)

    decoder_state = np.zeros([batch_size, constants.COLOR_SIZE], np.float32)
    decoder_input = np.zeros([batch_size, max_len - 1], np.int32)
    decoder_length = np.zeros([batch_size], np.int32)
    decoder_mask = np.zeros([batch_size, max_len - 1], np.float32)
    decoder_label = np.zeros([batch_size, max_len - 1], np.int32)

    for i, color in enumerate(batch):
        enc_name = [vocab[c] for c in color.name]

        encoder_input[i, :len(enc_name)] = enc_name
        encoder_length[i] = len(enc_name)
        encoder_target[i, :] = color.r, color.g, color.b

        decoder_state[i, :] = color.r, color.g, color.b
        decoder_input[i, :len(enc_name) - 1] = enc_name[:-1]
        decoder_length[i] = len(enc_name) - 1
        decoder_mask[i, :len(enc_name) - 1] = 1.0
        decoder_label[i, :len(enc_name) - 1] = enc_name[1:]

    return Batch(
        encoder_input,
        encoder_length,
        encoder_target,
        decoder_state,
        decoder_input,
        decoder_length,
        decoder_mask,
        decoder_label,
    )
