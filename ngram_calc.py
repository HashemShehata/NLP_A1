from collections import Counter
from data_tokenization import tokenize

def build_ngram(df, n):
    ngram_counts = Counter()

    for review in df:
        tokenized_sentences = tokenize(review,n)
        # print (tokenized_sentences)
        tokens_count = Counter(tuple(tokenized_sentences[i:i+n]) for i in range(len(tokenized_sentences) - n + 1))
        ngram_counts.update(tokens_count)
    return ngram_counts



def build_ngram_probabilities(ngram_counts,ngram_context_counts=None):
    ngram_probs = dict()

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
