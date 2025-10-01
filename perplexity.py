import math

def perplexity(probs, tokens_count):
    total_counts = 0
    logsum = 0
    for test_token, count in tokens_count.items():
        prob = probs.get(test_token)
        if prob <= 0:
            print (f"Test tokens are:",test_token)
            return float('inf')
        logsum+= count * math.log(prob)
        total_counts+= count
    return math.exp(-logsum / total_counts)