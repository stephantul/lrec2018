"""Experiment 2 in the paper."""
import numpy as np
import time

from tqdm import tqdm
from lrec2018.helpers import load_featurizers_ortho, \
                             normalize, \
                             filter_function_ortho, \
                             to_csv
from wordkit.readers import Celex, Lexique
from lexicon import read_blp_format, read_dlp_format, read_flp_format
from scipy.stats.stats import pearsonr
from somber import Som


if __name__ == "__main__":

    np.random.seed(44)

    corpora = (("Dutch", Celex, "../../corpora/celex/dpw.cd", read_dlp_format, "../../corpora/lexicon_projects/dlp2_items.tsv"),)

    lang, reader, path, lex_func, lex_path = corpora[0]

    fields = ('orthography', 'phonology', 'syllables', 'frequency')

    rt_data = dict(lex_func(lex_path))
    rt_data = {normalize(k): v for k, v in rt_data.items()}
    r = reader(path,
               fields=fields,
               merge_duplicates=True,
               filter_function=filter_function_ortho)

    words = r.transform()
    for x in words:
        x['orthography'] = normalize(x['orthography'])

    temp = set()
    new_words = []
    for x in words:
        if x['orthography'] not in rt_data:
            continue
        if x['orthography'] in temp:
            continue
        temp.add(x['orthography'])
        new_words.append(x)

    words = new_words

    ortho_forms = [x['orthography'] for x in words]
    featurizers, ids = zip(*load_featurizers_ortho(words))
    ids = list(ids)

    rt_values = np.asarray([rt_data[w['orthography']] for w in words])

    dists = []

    start = time.time()

    correlations = []

    indices = np.random.choice(np.arange(5000),
                               replace=False,)

    for idx, f in tqdm(enumerate(featurizers), total=len(featurizers)):
        print(idx)
        X = f.fit_transform(words).astype(np.float32)
        s = Som((20, 20), 1.0)
        s.fit(X, show_progressbar=True)
        err = s.quantization_error(X)
        correlations.append(pearsonr(err, rt_values)[0])
