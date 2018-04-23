import pandas as pd
import numpy as np

from collections import defaultdict
from itertools import combinations

# Alpha interval of .95
# corrected for multiple comparisons
Z_MULT = 2.98


def calculate_significance(array):
    """Calculate significance directly."""
    return is_sig(*interval_from_values(array))


def interval_from_values(array):
    """Calculate the interval directly from some values."""
    return interval(array.mean(), array.std(), Z_MULT)


def interval(mean, std, z):
    """Calculate the interval."""
    z_std = std * z
    return (mean - z_std, mean + z_std)


def is_sig(upper, lower):
    """See whether 0 is included in the confidence interval."""
    return not (upper < 0 < lower or upper > 0 > lower)


if __name__ == "__main__":

    df = pd.read_csv("/Users/stephantulkens/Google Drive/code/r/lrec/experiment_3_Dutch_words.csv")

    values = defaultdict(list)

    for x in range(1000):
        for y in df[df.iter == x].as_matrix():
            name = "{}-{}".format(y[0], y[1])
            values[name].append(y[2])

    values = {k: np.array(v) for k, v in values.items()}
    values_stat = {k: (v.mean(), v.std()) for k, v in values.items()}

    '''sig = {k: interval_from_values(np.array(v))
           for k, v in comparisons.items()}'''
