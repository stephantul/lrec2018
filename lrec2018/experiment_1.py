"""Experiments for the LREC 2018 paper."""
import numpy as np

from wordkit.readers import Celex
from wordkit.transformers import ONCTransformer, LinearTransformer, \
                                 OpenNGramTransformer, CVTransformer, \
                                 WickelTransformer, \
                                 ConstrainedOpenNGramTransformer, \
                                 WeightedOpenBigramTransformer
from wordkit.features import miikkulainen_features, \
                             miikkulainen, fourteen, \
                             sixteen, binary_features

from wordkit.feature_extraction import OneHotCharacterExtractor, \
                                       PhonemeFeatureExtractor, \
                                       PredefinedFeatureExtractor, \
                                       OneHotPhonemeExtractor

from scipy.stats.stats import pearsonr
from sklearn.pipeline import FeatureUnion

from reach import Reach
from itertools import product
from copy import deepcopy
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


def load_featurizers(words):
    """Load the featurizers for use in the experiments."""

    o_c = OneHotCharacterExtractor(field='orthography').extract(words)
    orthographic_features = {'miikkulainen': miikkulainen,
                             'fourteen': fourteen,
                             'sixteen': sixteen,
                             'one hot': o_c}
    m = PredefinedFeatureExtractor(miikkulainen_features,
                                   field='phonology').extract(words)
    b = PredefinedFeatureExtractor(binary_features,
                                   field='phonology').extract(words)

    o = OneHotPhonemeExtractor(field='phonology').extract(words)
    f = PhonemeFeatureExtractor(field='phonology').extract(words)

    phonological_features = {'one hot': o,
                             'features': f,
                             'miikkulainen features': m,
                             'binary features': b}

    inv_orthographic = {len(next(iter(v.values()))): k
                        for k, v in orthographic_features.items()}

    inv_phonological = {len(next(iter(v[1].values()))): k
                        for k, v in phonological_features.items()}

    possibles = []
    ids = []

    possible_ortho = list(product([LinearTransformer],
                          orthographic_features.values()))
    possible_ortho.append([OpenNGramTransformer, 0])
    possible_ortho.append([WickelTransformer, 0])
    possible_ortho.append([ConstrainedOpenNGramTransformer, 0])
    possible_ortho.append([WeightedOpenBigramTransformer, 0])
    possible_phono = list(product([CVTransformer, ONCTransformer],
                                  phonological_features.values()))
    possible_phono.append([OpenNGramTransformer, 0])
    possible_phono.append([WickelTransformer, 0])

    for ((o, o_f), (p, p_f)) in product(possible_ortho, possible_phono):
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

        if p == ONCTransformer:
            curr_p = p(p_f)
        elif p == LinearTransformer:

            temp = deepcopy(p_f[0])
            temp.update(p_f[1])
            length = len(next(iter(p_f[0].values())))
            if not all([len(x) == length for x in p_f[0]]):
                continue

            if not np.all(np.array(list(p_f.values())).sum(1) == 1):
                continue
            curr_p = p(p_f[0], field='phonology')
        elif p == OpenNGramTransformer:
            curr_p = p(n=2, field='phonology')
        elif p == WickelTransformer:
            curr_p = p(n=1, field='phonology')
        else:
            curr_p = p(p_f, field='phonology')

        possibles.append((("o", curr_o), ("p", curr_p)))

        try:
            p_len = len(next(iter(curr_p.features[1].values())))
        except (AttributeError, TypeError) as e:
            p_len = 0

        p_name = str(curr_p.__class__).split(".")[-1][:-2]

        if isinstance(curr_p, WickelTransformer):
            p_feat_name = p_name
        else:
            p_feat_name = inv_phonological[p_len]

        try:
            o_len = len(next(iter(curr_o.features.values())))
        except AttributeError:
            o_len = 0

        o_name = str(curr_o.__class__).split(".")[-1][:-2]
        if isinstance(curr_o, WickelTransformer):
            o_feat_name = o_name
        else:
            o_feat_name = inv_orthographic[o_len]

        ids.append((o_name,
                    p_name,
                    o_feat_name,
                    p_feat_name))

    for idx, f in enumerate(possibles):
        c = FeatureUnion(list(f))
        yield c, ids[idx]


if __name__ == "__main__":

    cel = Celex("../../corpora/celex/epl.cd",
                fields=('orthography', 'phonology', 'syllables'),
                language='eng',
                merge_duplicates=True,
                filter_function=filter_function_ortho)

    words = cel.transform()
    featurizers, ids = zip(*load_featurizers(words))

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
