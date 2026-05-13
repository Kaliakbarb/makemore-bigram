# makemore-bigram

A character-level bigram language model, trained on a list of names,
that samples new ones. The whole "model" is a Laplace-smoothed
table of `P(next char | current char)` - no gradient descent needed,
just counting.

    python cli.py names.txt 15

## What it actually captures

A bigram only ever looks one character back, so it learns things
like "names starting with 'z' are often followed by a vowel" but has
no memory of the name so far. Expect a mix of plausible-sounding
fragments and pure noise - that's the honest ceiling of a 1-gram-of-
context model, not a bug. Feeding it a bigger `names.txt` tightens
the samples up; going to trigrams or an MLP (see `tiny-autograd`)
tightens them up a lot more.
