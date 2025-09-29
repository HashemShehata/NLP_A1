import sys

def build_ngram_laplace_smoothing(ngram_counts,vocab_size,ngram_context_counts=None):
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
            ngram_probs[ngram] = (count+1) / (context_count+vocab_size)
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

# new code added by clowie: KN + stupid backoff smoothing
def build_kneser_ney_smoothing(ngram_counts, vocab_size, ngram_context_counts=None, discount=0.75):
    ngram_probs = dict()
    
    if ngram_context_counts is None:
        total_ngrams = sum(ngram_counts.values())
        context_count = total_ngrams
    
    for ngram, count in ngram_counts.items():
        context = ngram[:-1]
        if ngram_context_counts is not None:
            if isinstance(ngram_context_counts, dict):
                context_count = ngram_context_counts.get(context, 0)
        ngram_probs[ngram] = max(count - discount, 0) / context_count  # simplified KN
    return ngram_probs

def build_stupid_backoff(ngram_counts, ngram_context_counts, alpha=0.4):
    ngram_probs = dict()
    total_unigrams = sum([c for n, c in ngram_counts.items() if len(n) == 1])

    def sb_prob(ngram):
        context = ngram[:-1]
        # Check context only if ngram_context_counts is a dict
        if isinstance(ngram_context_counts, dict):
            context_count = ngram_context_counts.get(context, 0)
        else:
            context_count = 1  # dummy 1 to avoid zero division for unigrams

        if ngram_counts.get(ngram, 0) > 0 and context_count > 0:
            return ngram_counts[ngram] / context_count
        else:
            if len(ngram) == 1:
                # fallback for unigram
                return ngram_counts.get(ngram, 0) / sum(ngram_counts.values())
            else:
                # recursively backoff
                return alpha * sb_prob(ngram[1:])
            
# --- Kneser-Ney and Stupid Backoff Smoothing for Bigrams ---
def build_kneser_ney_bigram_probs(bigram_counts, unigram_counts, discount=0.75):
    # Collect continuation counts
    continuation_counts = Counter()
    for (w1, w2) in bigram_counts:
        continuation_counts[w2] += 1
    total_bigrams = sum(bigram_counts.values())
    total_unigrams = sum(unigram_counts.values())
    unique_bigrams = len(bigram_counts)
    unique_continuations = len(continuation_counts)

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
        p_cont = p_continuation.get(w2, 1e-8)
        if c_w1 > 0:
            return max(c_bigram - discount, 0) / c_w1 + lambda_w1 * p_cont
        else:
            return p_cont
    return kn_prob

def build_stupid_backoff_bigram_probs(bigram_counts, unigram_counts, alpha=0.4):
    total_unigrams = sum(unigram_counts.values())
    def sb_prob(w1, w2):
        bigram = (w1, w2)
        c_bigram = bigram_counts.get(bigram, 0)
        c_w1 = unigram_counts.get((w1,), 0)
        if c_bigram > 0 and c_w1 > 0:
            return c_bigram / c_w1
        else:
            # Backoff to unigram
            c_w2 = unigram_counts.get((w2,), 0)
            return alpha * (c_w2 / total_unigrams if total_unigrams > 0 else 1e-8)
    return sb_prob
