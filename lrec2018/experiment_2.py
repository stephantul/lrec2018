"""Experiment 2 in the paper."""
import json
import numpy as np
import time

from wordkit.readers import Celex
from tqdm import tqdm
from old20.old20 import old_subloop

from experiment_1 import load_featurizers
from lexicon import read_blp_format
from scipy.stats.stats import pearsonr


def select_from_blp(words, blp_path):
    """Select words from the blp."""
    blp = dict(read_blp_format(blp_path))
    for idx, word in enumerate(words):
        try:
            yield blp[word], idx
        except KeyError:
            pass


def filter_function_ortho(x):
    """Filter words based on punctuation and length."""
    a = not set(x['orthography']).intersection({' ', "'", '.', '/', ',', '-'})
    return (a and
            len(x['phonology']) < 12 and
            len(x['orthography']) < 10 and
            len(x['orthography']) >= 3)


if __name__ == "__main__":

    np.random.seed(44)

    rt_data = dict(read_blp_format("data/blp-items.txt"))

    cel = Celex("../../corpora/celex/epl.cd",
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

    featurizers, ids = zip(*load_featurizers(words))

    levenshtein_distances = old_subloop(ortho_forms, True)

    ids = list(ids)
    ids.append(("old_20", "old_20", "old_20", "old_20"))

    sample_results = []
    # Bootstrapping
    n_samples = 10
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
