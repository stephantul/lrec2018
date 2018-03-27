"""Experiments for the LREC 2018 paper."""
import numpy as np

from wordkit.readers import Celex

from scipy.stats.stats import pearsonr

from lrec2018.helpers import load_featurizers_combined
from reach import Reach
from tqdm import tqdm


def filter_function(allowed_phonemes, x):
    """Filter words based on punctuation and phonemes."""
    a = not set(x['orthography']).intersection({' ', "'", '.', '/', ',', '-'})
    b = not set(x['phonology']).difference(allowed_phonemes)

    return a and b and len(x['phonology']) < 12 and len(x['orthography']) < 10


def auto_distance(X, words):
    """Calculate the distance from a set of words to itself."""
    r = Reach(X, words)
    p = []

    for x in range(0, len(r.norm_vectors), 100):

        p.append(r.norm_vectors[x:x+100].dot(r.norm_vectors.T))

    p = np.concatenate(p)
    p[np.diag_indices(p.shape[0], 2)] = 0

    return p


if __name__ == "__main__":

    cel = Celex("../../corpora/celex/epl.cd",
                fields=('orthography', 'phonology', 'syllables'),
                language='eng',
                merge_duplicates=True,
                filter_function=filter_function)

    words = cel.transform()
    featurizers, ids = zip(*load_featurizers_combined(words))

    # Experiment 1
    from itertools import combinations

    scores = []
    old_20_scores = {}

    total = len(list(combinations(featurizers, 2)))

    buff = None

    wordlist = ["-".join([x['orthography'], "|".join(x['phonology']),
                          str(x['syllables'])])
                for x in words]

    all_matrices = np.zeros((len(featurizers), len(wordlist) * len(wordlist)))
    for idx, f in tqdm(enumerate(featurizers), total=len(featurizers)):
        X = f.fit_transform(words)
        X /= np.linalg.norm(X, 1)
        dist = X.dot(X.T)
        all_matrices[idx] += dist.flatten()

    print("Made matrix.")

    all_features = enumerate(combinations(np.arange(len(featurizers)), 2))
    for idx, (x, y) in tqdm(all_features, total=total):
        X, Y = all_matrices[x], all_matrices[y]
        scores.append(pearsonr(X.flatten(), Y.flatten()))
