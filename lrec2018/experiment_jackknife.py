"""Experiment 2 in the paper."""
import numpy as np

from tqdm import tqdm
from old20.old20 import old_subloop
from lrec2018.helpers import load_featurizers_ortho, \
                             normalize, \
                             filter_function_ortho, \
                             to_csv
from wordkit.readers import Subtlex, Lexique
from lexicon import read_blp_format, read_dlp_format, read_flp_format
from scipy.stats.stats import pearsonr, spearmanr


if __name__ == "__main__":

    np.random.seed(44)

    use_levenshtein = True

    corpora = (("nld", Subtlex, "../../corpora/subtlex/SUBTLEX-NL.cd-above2.txt", read_dlp_format, "../../corpora/lexicon_projects/dlp2_items.tsv"),
               ("eng-uk", Subtlex, "../../corpora/subtlex/SUBTLEX-UK.xlsx", read_blp_format, "../../corpora/lexicon_projects/blp-items.txt"),
               ("fra", Lexique, "../../corpora/lexique/Lexique382.txt", read_flp_format, "../../corpora/lexicon_projects/French Lexicon Project words.xls"))

    fields = ("orthography", "frequency")

    for lang, reader, path, lex_func, lex_path in corpora:

        rt_data = dict(lex_func(lex_path))
        rt_data = {normalize(k): v for k, v in rt_data.items()}
        r = reader(path,
                   language=lang,
                   fields=fields)

        words = r.transform(filter_function=filter_function_ortho)
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
        words = sorted(words,
                       key=lambda x: x['frequency'],
                       reverse=True)[:20000]

        ortho_forms = [x['orthography'] for x in words]

        if use_levenshtein:
            levenshtein_distances = old_subloop(ortho_forms, True)

        sample_results = []
        # Bootstrapping
        n_samples = 10000

        featurizers, ids = zip(*load_featurizers_ortho(words))
        ids = list(ids)

        estims = []

        if use_levenshtein:
            z = np.partition(levenshtein_distances, axis=1, kth=21)[:, :21]
            z = np.sort(z, 1)[:, 1:21].mean(1)
            ids = [("old_20", "old_20")] + ids
            estims.append(z)

        for idx, f in tqdm(enumerate(featurizers), total=len(featurizers)):

            X = f.fit_transform(words).astype(np.float32)
            X /= np.linalg.norm(X, axis=1)[:, None]
            x = 1 - X.dot(X.T)
            s = np.partition(x, axis=1, kth=21)[:, :21]
            s = np.sort(s, 1)[:, 1:21].mean(1)
            estims.append(s)

        for sample in tqdm(range(n_samples), total=n_samples):

            indices = np.random.choice(np.arange(len(ortho_forms)),
                                       size=len(words) * .8,
                                       replace=False)
            local_ortho = [ortho_forms[x] for x in indices]
            local_words = [words[x] for x in indices]
            rt_values = np.asarray([rt_data[w] for w in local_ortho])

            r = []

            for x in estims:

                s = x[indices]
                r.append((pearsonr(s, rt_values)[0],
                          spearmanr(s, rt_values)[0]))

            sample_results.append(r)

        sample_results = np.array(sample_results).transpose(1, 0, 2)
        to_csv("experiment_jackknife_{}_words.csv".format(lang),
               dict(zip(ids, sample_results)),
               ("pearson", "spearman"))
