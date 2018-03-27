"""Helpers for the experiments."""
import numpy as np
import unicodedata
from itertools import product
from copy import deepcopy
from sklearn.pipeline import FeatureUnion

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


def normalize(string):
    """Normalize, remove accents and other stuff."""
    s = unicodedata.normalize("NFKD", string).encode('ASCII', 'ignore')
    return s.decode('utf-8')


def to_csv(filename, data):
    """Write data to csv."""
    with open(filename, 'w') as f:
        f.write("o,o_f,RT\n")
        for k, v in data.items():
            for val in v:
                f.write("{},{},{}\n".format(k[0], k[1], val))


def load_featurizers_combined(words):
    """Load combined orthographic phonology featurizers."""
    ortho = load_featurizers_ortho(words)
    phono = load_featurizers_phono(words)

    for (o, o_id), (p, p_id) in product(ortho, phono):

        o_name, o_feat = o_id
        p_name, p_feat = p_id
        featurizer = FeatureUnion((("o", o), ("p", p)))
        yield featurizer, (o_name, p_name, o_feat, p_feat)


def load_featurizers_ortho(words):
    """Load the orthographic featurizers."""
    o_c = OneHotCharacterExtractor(field='orthography').extract(words)

    orthographic_features = {'miikkulainen': miikkulainen,
                             'fourteen': fourteen,
                             'sixteen': sixteen,
                             'one hot': o_c}

    possible_ortho = list(product([LinearTransformer],
                                  orthographic_features.items()))
    possible_ortho.append([OpenNGramTransformer, ("open ngrams", 0)])
    possible_ortho.append([WickelTransformer, ("wickelfeatures", 0)])
    possible_ortho.append([ConstrainedOpenNGramTransformer,
                          ("constrained bigrams", 0)])
    possible_ortho.append([WeightedOpenBigramTransformer,
                          ("weighted bigrams", 0)])

    for o, (f_name, o_f) in possible_ortho:
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

        yield curr_o, (o.__name__, f_name)


def load_featurizers_phono(words):
    """Load the phonological featurizers."""
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

    possible_phono = list(product([CVTransformer, ONCTransformer],
                                  phonological_features.items()))
    possible_phono.append([OpenNGramTransformer, ("open ngrams", 0)])
    possible_phono.append([WickelTransformer, ("wickelphones", 0)])

    for p, (f_name, p_f) in possible_phono:
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

        yield curr_p, (p.__name__, f_name)


def filter_function_ortho(x):
    """Filter words based on punctuation and length."""
    a = not set(x['orthography']).intersection({' ', "'", '.', '/', ',', '-'})
    return (a and
            len(x['phonology']) < 12 and
            len(x['orthography']) < 10 and
            len(x['orthography']) >= 3)
