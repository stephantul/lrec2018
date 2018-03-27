"""BLP functions."""
import pandas as pd


def read_blp_format(filename, words=set()):
    """Read RT data from the British Lexicon Project files."""
    words = set(words)
    f = open(filename)
    _ = next(f)
    for line in f:

        word, _, rt, *rest = line.strip().split("\t")

        if words and word not in words:
            continue
        try:
            yield((word, float(rt)))
        except ValueError:
            continue


def read_dlp_format(filename, words=set()):
    """Read RT data from the Dutch Lexicon Project files."""
    words = set(words)
    f = open(filename)
    _ = next(f)
    for line in f:

        _, _, word, _, _, _, rt, *rest = line.strip().split("\t")

        if words and word not in words:
            continue
        try:
            yield((word, float(rt)))
        except ValueError:
            continue


def read_flp_format(filename, words=set()):
    """Read RT data from Lexique."""
    words = set(words)
    for idx, line in pd.read_excel(filename).iterrows():
        if not words or line['item'] in words:
            yield((line['item'], line['rt']))
