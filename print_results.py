from perplexity import perplexity
from ngram_calc import build_ngram_probabilities
from smoothing import build_kneser_ney_bigram_probs, get_k_smoothing, build_stupid_backoff


def print_perplexity_results(ngram_probs, ngram_counts):
    print(f"Perplexity of validation set on bigrams (No Smoothing): {perplexity(ngram_probs, ngram_counts)}")

def print_no_smoothing_results(ngram_counts):
    ngram_probs = build_ngram_probabilities(ngram_counts)
    print_perplexity_results(ngram_probs, ngram_counts)

def print_k_smoothing_results(ngram_counts):
    for k in [0.01, 0.1, 0.5, 1.0]:
        vocab_size = len(ngram_counts)
        bigram_probs_k = get_k_smoothing(ngram_counts, k, vocab_size)
        print_perplexity_results(bigram_probs_k, ngram_counts)

def print_train_laplace_smoothing_results(ngram_counts, vocab_size):
    k = 1.0
    bigram_probs_laplace = get_k_smoothing(ngram_counts, k, vocab_size)
    print_perplexity_results(bigram_probs_laplace, ngram_counts)

def print_laplace_smoothing_results(ngram_counts, vocab_size):
    k = 1.0
    bigram_probs_laplace = get_k_smoothing(ngram_counts, k, vocab_size)
    print_perplexity_results(ngram_counts, bigram_probs_laplace)


def print_kn_results(train_bigram_counts, train_unigram_counts, bigram_counts, discount=0.75):
    kn_prob_func = build_kneser_ney_bigram_probs(train_bigram_counts, train_unigram_counts, discount=discount)
    bigram_probs_kn = {}
    for bigram in bigram_counts.keys():
        w1, w2 = bigram
        bigram_probs_kn[bigram] = kn_prob_func(w1, w2)
    print(f"Perplexity bigrams (KN): {perplexity(bigram_probs_kn, bigram_counts)}")



def print_stupid_backoff_results(train_bigram_counts, train_unigram_counts, bigram_counts, alpha=0.4):
    sb_prob = build_stupid_backoff(train_bigram_counts, train_unigram_counts, alpha=alpha)
    # Build probability dict only for evaluation bigrams
    eval_bigram_probs = {bg: sb_prob(bg[0], bg[1]) for bg in bigram_counts.keys()}

    perp = perplexity(eval_bigram_probs, bigram_counts)
    print(f"Perplexity of validation set on bigrams (SB, alpha={alpha}): {perp}")