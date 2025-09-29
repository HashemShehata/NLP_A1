import math

def calculate_perplexity(test_tokenized, ngram_probs, n, unk_token='<unk>'):
    N = 0
    log_prob_sum = 0.0
    is_func = callable(ngram_probs)
    for tokens in test_tokenized:
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            if is_func:
                # For Kneser-Ney or Stupid Backoff
                prob = ngram_probs(*ngram)
            else:
                prob = ngram_probs.get(ngram)
            if prob is None or prob == 0:
                if n == 1:
                    ngram = (unk_token,)
                    prob = ngram_probs.get(ngram, 1e-8) if not is_func else 1e-8
                else:
                    prob = 1e-8
            log_prob_sum += math.log(prob)
            N += 1
    perplexity = math.exp(-log_prob_sum / N) if N > 0 else float('inf')
    
    return perplexity

def perplexity(train_probs, test_tokens_count):
    total_counts = 0
    logsum = 0
    for test_token, count in test_tokens_count.items():
        prob = train_probs.get(test_token)
        if prob <= 0:
            print (f"Test tokens are:",test_token)
            return float('inf')
        logsum+= count * math.log(prob)
        total_counts+= count
    return math.exp(-logsum / total_counts)