"""Experiment 2 in the paper."""
import numpy as np
import time

from wordkit.readers import Celex
from wordkit.features import miikkulainen, fourteen, sixteen
from tqdm import tqdm
from old20.old20 import old_subloop

from lexicon import read_blp_format, read_dlp_format
from scipy.stats.stats import pearsonr

from wordkit.transformers import LinearTransformer, \
                                 OpenNGramTransformer, \
                                 WickelTransformer, \
                                 ConstrainedOpenNGramTransformer, \
                                 WeightedOpenBigramTransformer
from wordkit.feature_extraction import OneHotCharacterExtractor
from string import ascii_lowercase
from itertools import product
from lrec2018.experiment_2 import filter_function_ortho


def select_from_blp(words, blp_path):
    """Select words from the blp."""
    blp = dict(read_blp_format(blp_path))
    for idx, word in enumerate(words):
        try:
            yield blp[word], idx
        except KeyError:
            pass


z = OneHotCharacterExtractor().extract([ascii_lowercase])

orthographic_features = {'miikkulainen': miikkulainen,
                         'fourteen': fourteen,
                         'sixteen': sixteen,
                         'one hot': z}

inv_orthographic = {len(next(iter(v.values()))): k
                    for k, v in orthographic_features.items()}


def to_csv(filename, data):
    """Write data to csv."""
    with open(filename, 'w') as f:
        f.write("o,o_f,RT\n")
        for k, v in data.items():
            for val in v:
                f.write("{},{},{}\n".format(k[0], k[1], val))


def load_featurizers_ortho(words):
    """Load the orthographic featurizers."""
    o_c = OneHotCharacterExtractor(field='orthography').extract(words)
    orthographic_features = [fourteen,
                             sixteen,
                             o_c,
                             miikkulainen]

    possible_ortho = list(product([LinearTransformer], orthographic_features))
    possible_ortho.append([OpenNGramTransformer, 0])
    possible_ortho.append([WickelTransformer, 0])
    possible_ortho.append([ConstrainedOpenNGramTransformer, 0])
    possible_ortho.append([WeightedOpenBigramTransformer, 0])

    possibles = []
    ids = []

    for (o, o_f) in possible_ortho:
        if o == LinearTransformer:
            curr_o = o(o_f, field='orthography')
        elif o == WickelTransformer:
            curr_o = o(n=3, field='orthography')
        elif o == OpenNGramTransformer:
            curr_o = o(n=2, field='orthography')
        elif o == ConstrainedOpenNGramTransformer:
            curr_o = o(n=2, window=3, field='orthography')
        elif o == WeightedOpenBigramTransformer:
            curr_o = o(weights=(1, .7, .2), field='orthography')
        else:
            curr_o = o(field='orthography')

        possibles.append(curr_o)
        try:
            o_len = len(next(iter(curr_o.features.values())))
        except AttributeError:
            o_len = 0

        o_name = str(curr_o.__class__).split(".")[-1][:-2]
        if o_name in ["WickelTransformer",
                      "OpenNGramTransformer",
                      "ConstrainedOpenNGramTransformer",
                      "WeightedOpenBigramTransformer"]:
            o_feat_name = o_name
        else:
            o_feat_name = inv_orthographic[o_len]

        ids.append((o_name,
                    o_feat_name))

    for idx, f in enumerate(possibles):
        yield f, ids[idx]


if __name__ == "__main__":

    np.random.seed(44)

    rt_data = dict(read_blp_format("data/blp-items.txt"))

    cel = Celex("../../corpora/celex/epw.cd",
                fields=('orthography', 'phonology', 'syllables'),
                language='eng',
                merge_duplicates=True,
                filter_function=filter_function_ortho)

    words = cel.transform(set(rt_data.keys()))

    temp = set()
    new_words = []
    for x in words:
        if x['orthography'] in temp:
            continue
        temp.add(x['orthography'])
        new_words.append(x)
    words = new_words

    ortho_forms = [x['orthography'] for x in words]

    featurizers, ids = zip(*load_featurizers_ortho(words))

    levenshtein_distances = old_subloop(ortho_forms, True)

    ids = list(ids)
    ids.append(("old_20", "old_20"))

    sample_results = []
    # Bootstrapping
    n_samples = 100
    values_to_test = (20,)
    for sample in tqdm(range(n_samples), total=n_samples):

        indices = np.random.choice(np.arange(len(ortho_forms)),
                                   size=9000,
                                   replace=False)
        local_ortho = [ortho_forms[x] for x in indices]
        local_words = [words[x] for x in indices]
        rt_values = np.asarray([rt_data[w] for w in local_ortho])

        dists = []

        start = time.time()
        o = np.sort(levenshtein_distances[indices][:, indices], 1)

        for idx, f in tqdm(enumerate(featurizers), total=len(featurizers)):
            X = f.fit_transform(local_words).astype(np.float32)
            X /= np.linalg.norm(X, axis=1)[:, None]
            dists.append(1 - X.dot(X.T))

        dists.append(o)

        r = []
        for x in dists:
            t = []
            for val in values_to_test:
                t.append(pearsonr(np.sort(x, 1)[:, 1:val+1].mean(1),
                         rt_values)[0])
            r.append(t)

        sample_results.append(r)

        print("Sample {} took {} seconds".format(sample,
                                                 time.time() - start))

    sample_results = np.squeeze(sample_results).T
    to_csv("experiment_3_english_words.csv", dict(zip(ids, sample_results)))
