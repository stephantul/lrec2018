"""Experiment 2 in the paper."""
import numpy as np
import time

from tqdm import tqdm
from old20.old20 import old_subloop
from lrec2018.helpers import load_featurizers_phono, \
                             normalize, \
                             filter_function_ortho, \
                             to_csv
from wordkit.readers import Celex, Lexique
from lexicon import read_blp_format, read_dlp_format, read_flp_format
from scipy.stats.stats import pearsonr


if __name__ == "__main__":

    np.random.seed(44)

    corpora = (("French", Lexique, "../../corpora/lexique/Lexique382.txt", read_flp_format, "../../corpora/lexicon_projects/French Lexicon Project words.xls"),
               ("Dutch", Celex, "../../corpora/celex/dpw.cd", read_dlp_format, "../../corpora/lexicon_projects/dlp2_items.tsv"),
               ("English", Celex, "../../corpora/celex/epw.cd", read_blp_format, "../../corpora/lexicon_projects/blp-items.txt"))

    fields = ("orthography", "phonology", "syllables")

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
        phono_forms = ["".join(x['phonology']) for x in words]
        featurizers, ids = zip(*load_featurizers_phono(words))
        ids = list(ids)
        ids.append(("old_20", "old_20"))
        ids.append(("pld_20", "pld_20"))

        levenshtein_distances_orth = old_subloop(ortho_forms, True)
        levenshtein_distances_phon = old_subloop(phono_forms, True)

        sample_results = []
        # Bootstrapping
        n_samples = 1000
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
            o = np.sort(levenshtein_distances_orth[indices][:, indices], 1)
            p = np.sort(levenshtein_distances_phon[indices][:, indices], 1)

            for idx, f in tqdm(enumerate(featurizers), total=len(featurizers)):
                X = f.fit_transform(local_words).astype(np.float32)
                X /= np.linalg.norm(X, axis=1)[:, None]
                dists.append(1 - X.dot(X.T))

            dists.append(o)
            dists.append(p)

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
        to_csv("experiment_4_{}_words.csv".format(lang),
               dict(zip(ids, sample_results)))
