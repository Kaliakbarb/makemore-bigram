import random
import sys

from bigram import build_bigram_counts, counts_to_probs, load_names, negative_log_likelihood, sample_name


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "names.txt"
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    names = load_names(path)
    counts = build_bigram_counts(names)
    probs = counts_to_probs(counts)

    print(f"trained on {len(names)} names, avg NLL {negative_log_likelihood(probs, names):.3f}")
    rng = random.Random()
    seen = set(names)
    for _ in range(count):
        name = sample_name(probs, rng)
        tag = " (new)" if name and name not in seen else ""
        print(f"  {name}{tag}")


if __name__ == "__main__":
    main()
