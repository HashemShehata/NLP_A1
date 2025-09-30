from collections import Counter
from data_tokenization import tokenize
import sys
from unk_handling import train_tokenized_unk



def build_ngram(df, n):
    ngram_counts = Counter()

    for review in df:
        tokenized_sentences = tokenize(review,n)
        # print (tokenized_sentences)
        tokens_count = Counter(tuple(tokenized_sentences[i:i+n]) for i in range(len(tokenized_sentences) - n + 1))
        ngram_counts.update(tokens_count)
    return ngram_counts



def build_ngram_probabilities(ngram_counts,ngram_context_counts=None, prob_vocab=None):
    ngram_probs = dict()
    ### for testing or validation get probs directly
    if prob_vocab is not None:
        for ngram,_ in ngram_counts.items():
            ngram_probs[ngram] = prob_vocab.get(ngram,0)
        return ngram_probs
    ### for calculating training probs
    if ngram_context_counts is None:
        total_ngrams = sum(ngram_counts.values())
        context_count = total_ngrams
    for ngram, count in ngram_counts.items():
        context = ngram[:-1]
        if ngram_context_counts is not None: 
            context_count = ngram_context_counts.get(context, 0)
        try:    
            ngram_probs[ngram] = count / context_count
        except Exception as e:
            print (e)
            print (f"Context words are {context}")
            sys.exit(0)
    return ngram_probs   


# Rebuild ngram counts with <unk> using tokenized lists
def build_ngram_from_tokenized(tokenized_reviews, n):
    ngram_counts = Counter()
    for tokens in tokenized_reviews:
        ngram_counts.update(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
    return ngram_counts


