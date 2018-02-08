import numpy as np
import json

from matplotlib import pyplot as plt
from itertools import combinations
from wordkit.features import patpho_bin, patpho_real, miikkulainen_features, plunkett_phonemes, miikkulainen, fourteen, sixteen, binary_features
from wordkit.feature_extraction import phoneme_features, one_hot_characters, one_hot_phoneme_features, one_hot_phonemes
from string import ascii_lowercase


orthographic_features = {'miikkulainen': miikkulainen,
                         'fourteen': fourteen,
                         'sixteen': sixteen,
                         'one hot': one_hot_characters(ascii_lowercase)}

phonological_features = {'patpho bin': patpho_bin,
                         'patpho real': patpho_real,
                         'plunkett': plunkett_phonemes,
                         'one hot': one_hot_phonemes(),
                         'one hot features': one_hot_phoneme_features(),
                         'miikkulainen features': phoneme_features(miikkulainen_features,
                                                                   use_is_vowel=False),
                         'binary features': phoneme_features(binary_features,
                                                             use_is_vowel=False)}

inv_orthographic = {len(next(iter(v.values()))): k
                    for k, v in orthographic_features.items()}

inv_phonological = {len(next(iter(v[1].values()))): k
                    for k, v in phonological_features.items()}


if __name__ == "__main__":

    scores = np.load(open("scores_experiment1_lrec_2018.npy", "rb"))
    ids = json.load(open("overlap_ids.json"))
    new_ids = []
    for n in ids:
        if n[2] == "WickelTransformer":
            n[2] = "wickel"
            n[0] = "w"
        elif n[2] == "OpenNGramTransformer":
            n[2] = "ngram"
            n[0] = "n"
        else:
            n[2] = n[2].split()[0]

        if n[3] == "WickelTransformer":
            n[3] = "wickel"
            n[1] = "w"
        elif n[3] == "OpenNGramTransformer":
            n[3] = "ngram"
            n[1] = "n"
        else:
            n[3] = n[3].split()[0]

        new_ids.append(n)

    ids = new_ids

    scores = np.array(scores)

    mtr = np.zeros((len(ids), len(ids)))
    mtr[np.diag_indices(len(ids), 2)] = 1

    for score, (x, y) in zip(scores, combinations(np.arange(len(ids)), 2)):

        mtr[x, y] = score
        mtr[y, x] = score

    coords = []
    for num in range(0, 60, 10):
        for z in [0, 4, 8, 9]:
            coords.append(num + z)

    f = plt.figure(figsize=(10, 10))
    a = ["\n{}".format(ids[x][2]) for x in range(0, 60, 10)]
    b = ["{}".format(ids[x][1])[:2].lower() for x in coords]
    plt.xticks(np.array(list(range(0, 60, 10)) + coords)-.5,
               a + b,
               ha='left')

    '''names = ['patpho_bin', 'patpho_real', 'plunkett', 'one-hot', 'one-hot features', 'miikkulainen', 'binary features']
    ticks = np.arange(len(ids))
    nnames = []
    for x in ticks:
        nnames.append(names[x % len(names)])'''

    plt.yticks([], [], va='center')
    plt.tick_params(labelsize=10, length=0, axis='x')
    plt.tick_params(labelsize=7.5, length=0, axis='y')
    plt.imshow(mtr, cmap='viridis', vmin=.42, vmax=1.0)
    plt.savefig("correlation_plot_no_y.png", bbox_inches='tight', pad_inches=0)
