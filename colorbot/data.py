"""Module related to color data and samples.
"""
from collections import namedtuple

Color = namedtuple("Color", ("name", "r", "g", "b"))


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
