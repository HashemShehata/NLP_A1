from perplexity import perplexity
from ngram_calc import build_ngram_probabilities
from smoothing import build_kneser_ney_bigram_probs, get_k_smoothing, build_stupid_backoff,get_k_smoothing


def print_perplexity_results(ngram_probs, ngram_counts):
    print(f"Perplexity score: {perplexity(ngram_probs, ngram_counts)}")

def print_no_smoothing_results(ngram_counts, ngram_context_counts=None, prob_vocab=None):
    ngram_probs = build_ngram_probabilities(ngram_counts, ngram_context_counts, prob_vocab)
    print_perplexity_results(ngram_probs, ngram_counts)

def print_k_smoothing_results(val_ngram_counts,k,vocab_size,train_ngram_counts,train_ngram_context_counts=None,train_prob_vocab=None):
    val_ngram_probs = get_k_smoothing(val_ngram_counts,k,vocab_size,train_ngram_counts,\
                                      train_ngram_context_counts=train_ngram_context_counts,\
                                        train_prob_vocab=train_prob_vocab)
    print_perplexity_results(val_ngram_probs, val_ngram_counts)

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