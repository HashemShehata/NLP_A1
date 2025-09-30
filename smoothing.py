import sys
from collections import Counter
def get_k_smoothing(val_ngram_counts,k,vocab_size,train_ngram_counts,train_ngram_context_counts=None,train_prob_vocab=None):
    ngram_probs = dict()
    for ngram, _ in val_ngram_counts.items():
        prob_ngram = train_prob_vocab.get(ngram,0)
        if prob_ngram!=0:
            ngram_probs[ngram]=prob_ngram
            continue
        
        context = ngram[:-1]
        if train_ngram_context_counts is not None: 
            context_count = train_ngram_context_counts.get(context, 0)
        else:
            # For unigrams, use total corpus size as context
            total_ngrams = sum(train_ngram_counts.values())
            context_count = total_ngrams
        try:
            count=0
            ngram_probs[ngram] = (count+k) / (context_count+k*vocab_size)
        except Exception as e:
            print (e)
            print (f"Context words are {context}")
            sys.exit(0)
    return ngram_probs


def build_k_smoothing(ngram_counts,k,vocab_size,ngram_context_counts=None):
    ngram_probs = dict()
    for ngram, count in ngram_counts.items():
        context = ngram[:-1]
        if ngram_context_counts is not None: 
            context_count = ngram_context_counts.get(context, 0)
        else:
            # For unigrams, use total corpus size as context
            total_ngrams = sum(ngram_counts.values())
            context_count = total_ngrams
        try:
            ngram_probs[ngram] = (count+k) / (context_count+k*vocab_size)
        except Exception as e:
            print (e)
            print (f"Context words are {context}")
            sys.exit(0)
    return ngram_probs
            
# --- Kneser-Ney and Stupid Backoff Smoothing for Bigrams ---
def build_kneser_ney_bigram_probs(bigram_counts, unigram_counts, discount=0.75):
    # Collect continuation counts
    continuation_counts = Counter()
    for (w1, w2) in bigram_counts:
        continuation_counts[w2] += 1
    unique_bigrams = len(bigram_counts)

    # Precompute lambdas for each context
    lambdas = {}
    for (w1,) in unigram_counts:
        n_continuations = len([w2 for (ww1, w2) in bigram_counts if ww1 == w1])
        lambdas[w1] = (discount * n_continuations) / unigram_counts[(w1,)] if unigram_counts[(w1,)] > 0 else 0.0

    # Precompute continuation probabilities
    p_continuation = {w2: continuation_counts[w2] / unique_bigrams for w2 in continuation_counts}

    def kn_prob(w1, w2):
        bigram = (w1, w2)
        c_bigram = bigram_counts.get(bigram, 0)
        c_w1 = unigram_counts.get((w1,), 0)
        lambda_w1 = lambdas.get(w1, 0.0)
        p_cont = p_continuation.get(w2,0)
        if c_w1 > 0:
            return max(c_bigram - discount, 0) / c_w1 + lambda_w1 * p_cont
        else:
            return p_cont
    return kn_prob

def build_stupid_backoff(bigram_counts, unigram_counts, alpha=0.4):
    total_unigrams = sum(unigram_counts.values())
    def sb_prob(w1, w2):
        c_bigram = bigram_counts.get((w1, w2), 0)
        c_w1 = unigram_counts.get((w1,), 0)
        if c_bigram > 0 and c_w1 > 0:
            prob = c_bigram / c_w1
            return prob
        # backoff path
        c_w2 = unigram_counts.get((w2,), 0)
        prob = alpha * (c_w2 / total_unigrams)
        return prob
    return sb_prob
