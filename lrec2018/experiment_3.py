"""Experiment 2 in the paper."""
import numpy as np

from tqdm import tqdm
from old20.old20 import old_subloop
from lrec2018.helpers import load_featurizers_ortho, \
                             normalize, \
                             filter_function_ortho, \
                             to_csv
from wordkit.readers import Celex, Lexique
from lexicon import read_blp_format, read_dlp_format, read_flp_format
from scipy.stats.stats import pearsonr, spearmanr


if __name__ == "__main__":

    np.random.seed(44)

    use_levenshtein = False

    corpora = (("Dutch", Celex, "../../corpora/celex/dpw.cd", read_dlp_format, "../../corpora/lexicon_projects/dlp2_items.tsv"),
               ("English", Celex, "../../corpora/celex/epw.cd", read_blp_format, "../../corpora/lexicon_projects/blp-items.txt"),
               ("French", Lexique, "../../corpora/lexique/Lexique382.txt", read_flp_format, "../../corpora/lexicon_projects/French Lexicon Project words.xls"))

    fields = ("orthography", "phonology")

    for lang, reader, path, lex_func, lex_path in corpora:

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

        if use_levenshtein:
            levenshtein_distances = old_subloop(ortho_forms, True)

        sample_results = []
        # Bootstrapping
        n_samples = 1000
        values_to_test = (20,)
        for sample in tqdm(range(n_samples), total=n_samples):

            indices = np.random.choice(np.arange(len(ortho_forms)),
                                       size=len(words),
                                       replace=True)
            local_ortho = [ortho_forms[x] for x in indices]
            local_words = [words[x] for x in indices]
            rt_values = np.asarray([rt_data[w] for w in local_ortho])

            r = []
            featurizers, ids = zip(*load_featurizers_ortho(words))
            ids = list(ids)

            if use_levenshtein:
                l = levenshtein_distances[indices][:, indices]
                z = np.partition(l, axis=1, kth=21)[:, :21]
                z = np.sort(z, 1)[:, 1:21].mean(1)
                r.append(pearsonr(z, rt_values)[0])
                ids = [("old_20", "old_20")] + ids

            for idx, f in tqdm(enumerate(featurizers), total=len(featurizers)):

                X = f.fit_transform(local_words).astype(np.float32)
                X /= np.linalg.norm(X, axis=1)[:, None]
                x = 1 - X.dot(X.T)

                s = np.partition(x, axis=1, kth=21)[:, :21]
                s = np.sort(s, 1)[:, 1:21].mean(1)
                r.append((pearsonr(s, rt_values)[0],
                          spearmanr(s, rt_values)[0]))

            print(r)
            sample_results.append(r)

        sample_results = np.array(sample_results).transpose(1, 0, 2)
        to_csv("experiment_3_{}_words.csv".format(lang),
               dict(zip(ids, sample_results)),
               ("pearson", "spearman"))
