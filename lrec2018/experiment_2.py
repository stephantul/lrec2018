"""Experiment 2 in the paper."""
import json
import numpy as np
import time

from wordkit.readers import Celex
from wordkit.features import miikkulainen_features, binary_features
from wordkit.feature_extraction import phoneme_features, \
                                       one_hot_phoneme_features, \
                                       one_hot_phonemes
from tqdm import tqdm
from old20 import old20

from functools import partial
from experiment_1 import filter_function, load_featurizers
from blp import read_blp_format
from scipy.stats.stats import pearsonr


def select_from_blp(words, blp_path):
    """Select words from the blp."""
    blp = dict(read_blp_format(blp_path))
    for idx, word in enumerate(words):
        try:
            yield blp[word], idx
        except KeyError:
            pass


if __name__ == "__main__":

    phonological_features = [one_hot_phonemes(),
                             one_hot_phoneme_features(),
                             phoneme_features(miikkulainen_features,
                                              use_is_vowel=False),
                             phoneme_features(binary_features,
                                              use_is_vowel=False)]

    # only include words that are possible in all sets of phonemes.
    # therefore, find the intersection of all sets of phonemes.
    allowed_phonemes = set()
    for v, c in phonological_features:

        feats = set(v.keys()).union(c.keys())
        if not allowed_phonemes:
            allowed_phonemes = feats
        else:
            allowed_phonemes.intersection_update(feats)

    cel = Celex("../../corpora/celex/epl.cd",
                fields=('orthography', 'phonology', 'syllables'),
                language='eng',
                merge_duplicates=True,
                filter_function=partial(filter_function, allowed_phonemes))

    blp = dict(read_blp_format("data/blp-items.txt"))
    words_ = [x['orthography'] for x in json.load(open("lrec_2018_words.json"))
              if x['orthography'] in blp]
    words = cel.transform(words_)

    temp = set()
    new_words = []
    for x in words:
        if x['orthography'] in temp:
            continue
        temp.add(x['orthography'])
        new_words.append(x)
    words = new_words

    ortho_forms = [x['orthography'] for x in words]
    featurizers, ids = zip(*load_featurizers(words))

    sample_results = []
    # Bootstrapping
    n_samples = 100
    for sample in tqdm(range(n_samples), total=n_samples):

        indices = np.random.choice(np.arange(len(ortho_forms)), size=9000)
        local_ortho = [ortho_forms[x] for x in indices]
        local_words = [words[x] for x in indices]
        blp_values = np.asarray([blp[w] for w in local_ortho])

        o = old20(local_ortho)

        dists = []

        start = time.time()

        for idx, f in enumerate(featurizers):
            X = f.fit_transform(local_words)
            sums = X.sum(1)
            # broadcast into square
            sums = sums + sums[:, None]
            dist = sums - X.dot(X.T) * 2
            dists.append(dist)

        r = []
        for x in dists:
            r.append(pearsonr(np.sort(x, 1)[:, 1:21].mean(1), blp_values)[0])

        print("Sample {} took {} seconds".format(sample,
                                                 time.time() - start))

        old_20_values = np.asarray([o[w] for w in local_ortho])
        r.append(pearsonr(old_20_values, blp_values)[0])
        sample_results.append(r)

    ids.append(("old_20", "old_20", "old_20", "old_20"))
