

def read_blp_format(filename, words=set()):

    words = set(words)
    f = open(filename)
    _ = next(f)
    for line in f:

        word, _, rt, *rest = line.strip().split("\t")

        if words and word not in words:
            continue
        try:
            yield((word, float(rt)))
        except ValueError:
            continue
