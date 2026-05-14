"""Character-level bigram language model over a list of names."""
import math


def load_names(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]


def build_bigram_counts(names):
    counts = {}
    for name in names:
        chars = ["."] + list(name) + ["."]
        for a, b in zip(chars, chars[1:]):
            counts.setdefault(a, {})
            counts[a][b] = counts[a].get(b, 0) + 1
    return counts


def counts_to_probs(counts, smoothing=0.2):
    alphabet = set(counts.keys())
    for row in counts.values():
        alphabet |= set(row.keys())
    probs = {}
    for a in alphabet:
        row = counts.get(a, {})
        total = sum(row.values()) + smoothing * len(alphabet)
        probs[a] = {b: (row.get(b, 0) + smoothing) / total for b in alphabet}
    return probs


def sample_name(probs, rng, max_len=20):
    out, ch = [], "."
    for _ in range(max_len):
        row = probs.get(ch)
        if not row:
            break
        chars, weights = zip(*row.items())
        ch = rng.choices(chars, weights=weights)[0]
        if ch == ".":
            break
        out.append(ch)
    return "".join(out)


def negative_log_likelihood(probs, names):
    nll, n = 0.0, 0
    for name in names:
        chars = ["."] + list(name) + ["."]
        for a, b in zip(chars, chars[1:]):
            p = probs.get(a, {}).get(b, 1e-6)
            nll += -math.log(p)
            n += 1
    return nll / n
