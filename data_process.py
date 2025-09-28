
import pandas as pd
from collections import defaultdict,Counter
import nltk
nltk.download('punkt_tab')
import random
random.seed(42)
import sys
import math

def read_file(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            reviews = [line.strip() for line in f.readlines()]
            return reviews
    except Exception as e:
        print (f"Unable to read the file {filename}: {e}")
        return 
    
def n_count(tokens, n):
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

def tokenize(text,n):

    sentences = nltk.sent_tokenize(text)
    start_sentence = ['<s>']
    tokenized_sentences = []

    for sentence in sentences:
        processed_sentence = sentence.lower()
        start_padding = max(1, n - 1)
        tokens = nltk.word_tokenize(processed_sentence)
        final_tokens = start_sentence*start_padding + tokens + ['</s>']
        tokenized_sentences.extend(final_tokens)

    return tokenized_sentences


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

def build_ngram_laplace_smoothing(ngram_counts,vocab_size,ngram_context_counts=None):
    ngram_probs = dict()

    if ngram_context_counts is None:
        total_ngrams = sum(ngram_counts.values())
        context_count = total_ngrams
    for ngram, count in ngram_counts.items():
        context = ngram[:-1]
        if ngram_context_counts is not None: 
            context_count = ngram_context_counts.get(context, 0)
        try:    
            ngram_probs[ngram] = (count+1) / (context_count+vocab_size)
        except Exception as e:
            print (e)
            print (f"Context words are {context}")
            sys.exit(0)
    return ngram_probs  

def build_k_smoothing(ngram_counts,k,vocab_size,ngram_context_counts=None):
    ngram_probs = dict()

    if ngram_context_counts is None:
        total_ngrams = sum(ngram_counts.values())
        context_count = total_ngrams
    for ngram, count in ngram_counts.items():
        context = ngram[:-1]
        if ngram_context_counts is not None: 
            context_count = ngram_context_counts.get(context, 0)
        try:    
            ngram_probs[ngram] = (count+k) / (context_count+k*vocab_size)
        except Exception as e:
            print (e)
            print (f"Context words are {context}")
            sys.exit(0)
    return ngram_probs  

def compare_dicts(train_dict,val_dict):
    keys_not_in_train = set(val_dict.keys()) - set(train_dict.keys())
    return keys_not_in_train


# --- Handle unknowns: replace all tokens with freq=1 in training data with <unk> ---
train_set = read_file("A1_DATASET/train.txt")
random.shuffle(train_set)

# Split train/val
val_ratio = 0.2  
split_index = int(len(train_set) * val_ratio)
val_df = train_set[:split_index]
train_df = train_set[split_index:]

# First pass: get unigram counts
raw_unigram_counts = build_ngram(train_df, 1)

# Find tokens with freq=1
rare_tokens = set([token for token, count in raw_unigram_counts.items() if count == 2])



# --- Tokenization mode switch ---
TOKENIZATION_MODE = 'byte'  # 'word' or 'byte'

def byte_tokenize(text, n):
    # Tokenize text into bytes, return list of byte strings
    text_bytes = text.encode('utf-8')
    # For n-gram, treat each byte as a token (as int or as byte string)
    return [str(b) for b in text_bytes]

def tokenize_switch(text, n):
    if TOKENIZATION_MODE == 'word':
        return tokenize(text, n)
    elif TOKENIZATION_MODE == 'byte':
        return byte_tokenize(text, n)
    else:
        raise ValueError('Unknown tokenization mode')

# Replace rare tokens in tokenized lists, not as joined strings
def replace_rare_with_unk_tokenized(sentences, rare_tokens, n):
    tokenized_reviews = []
    for review in sentences:
        tokens = tokenize_switch(review, n)
        new_tokens = ["<unk>" if (token,) in rare_tokens else token for token in tokens]
        tokenized_reviews.append(new_tokens)
    return tokenized_reviews


# Replace rare tokens in train_df for unigrams (tokenized)
train_tokenized_unk = replace_rare_with_unk_tokenized(train_df, rare_tokens, 1)

# Rebuild ngram counts with <unk> using tokenized lists
def build_ngram_from_tokenized(tokenized_reviews, n):
    ngram_counts = Counter()
    for tokens in tokenized_reviews:
        ngram_counts.update(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
    return ngram_counts

unigram_counts = build_ngram_from_tokenized(train_tokenized_unk, 1)
bigram_counts = build_ngram_from_tokenized(train_tokenized_unk, 2)
# trigram_counts = build_ngram_from_tokenized(train_tokenized_unk, 3)

vocabulary_size = len(set(unigram_counts.keys()))
print(vocabulary_size)


# Output unigram counts
sorted_unigram_counter = sorted(unigram_counts.items(), key=lambda x: x[1], reverse=True)
with open("output_unigram_counts.txt", "w") as f:
    for item in sorted_unigram_counter:
        f.write(f"{item[0]}: {item[1]}\n")

# Output bigram counts
sorted_bigram_counter = sorted(bigram_counts.items(), key=lambda x: x[1], reverse=True)
with open("output_bigram_counts.txt", "w") as f:
    for item in sorted_bigram_counter:
        f.write(f"{item[0]}: {item[1]}\n")



# --- Perplexity calculation functions ---
import math

def calculate_perplexity(test_tokenized, ngram_probs, n, unk_token='<unk>'):
    N = 0
    log_prob_sum = 0.0
    for tokens in test_tokenized:
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            prob = ngram_probs.get(ngram)
            if prob is None or prob == 0:
                # If ngram not found, use <unk> for unigram, or assign small prob for bigram
                if n == 1:
                    ngram = (unk_token,)
                    prob = ngram_probs.get(ngram, 1e-8)
                else:
                    prob = 1e-8
            log_prob_sum += math.log(prob)
            N += 1
    perplexity = math.exp(-log_prob_sum / N) if N > 0 else float('inf')
    return perplexity

def tokenize_reviews_for_eval(reviews, n):
    return [tokenize_switch(review, n) for review in reviews]

# Read test set before tokenization
test_df = read_file("A1_DATASET/val.txt")


# Build set of known tokens (training vocab, after <unk> replacement)
train_vocab = set(token for (token,), count in unigram_counts.items())

# Function to replace OOV tokens with <unk> in tokenized data
def replace_oov_with_unk(tokenized_reviews, train_vocab):
    return [[token if token in train_vocab else '<unk>' for token in tokens] for tokens in tokenized_reviews]


# Tokenize validation and test sets
val_tokenized = tokenize_reviews_for_eval(val_df, 1)
val_tokenized_bigram = tokenize_reviews_for_eval(val_df, 2)
test_tokenized = tokenize_reviews_for_eval(test_df, 1)
test_tokenized_bigram = tokenize_reviews_for_eval(test_df, 2)

# # Diagnostic: count OOV tokens before <unk> replacement
# def count_oov(tokenized_reviews, train_vocab):
#     return sum(1 for tokens in tokenized_reviews for token in tokens if token not in train_vocab)

# val_oov_count = count_oov(val_tokenized, train_vocab)
# test_oov_count = count_oov(test_tokenized, train_vocab)
# print(f"OOV tokens in validation set before <unk> replacement: {val_oov_count}")
# print(f"OOV tokens in test set before <unk> replacement: {test_oov_count}")


# # Replace OOV tokens with <unk> in val/test sets
# val_tokenized = replace_oov_with_unk(val_tokenized, train_vocab)
# test_tokenized = replace_oov_with_unk(test_tokenized, train_vocab)

# # Diagnostic: count <unk> tokens after replacement
# def count_unk(tokenized_reviews):
#     return sum(1 for tokens in tokenized_reviews for token in tokens if token == '<unk>')

# val_unk_count = count_unk(val_tokenized)
# test_unk_count = count_unk(test_tokenized)
# print(f"<unk> tokens in validation set after replacement: {val_unk_count}")
# print(f"<unk> tokens in test set after replacement: {test_unk_count}")


# --- Perplexity calculations ---
print("\n--- Perplexity Results ---")

# No smoothing
unigram_probs_nosmooth = build_ngram_probabilities(unigram_counts)
bigram_probs_nosmooth = build_ngram_probabilities(bigram_counts, unigram_counts)
val_perp_uni_nosmooth = calculate_perplexity(val_tokenized, unigram_probs_nosmooth, 1)
val_perp_bi_nosmooth = calculate_perplexity(val_tokenized_bigram, bigram_probs_nosmooth, 2)
print(f"Validation Unigram Perplexity (no smoothing): {val_perp_uni_nosmooth:.2f}")
print(f"Validation Bigram Perplexity (no smoothing): {val_perp_bi_nosmooth:.2f}")

# Laplace smoothing
unigram_probs_laplace = build_ngram_laplace_smoothing(unigram_counts, vocabulary_size)
bigram_probs_laplace = build_ngram_laplace_smoothing(bigram_counts, vocabulary_size, unigram_counts)
val_perp_uni_laplace = calculate_perplexity(val_tokenized, unigram_probs_laplace, 1)
val_perp_bi_laplace = calculate_perplexity(val_tokenized_bigram, bigram_probs_laplace, 2)
print(f"Validation Unigram Perplexity (Laplace): {val_perp_uni_laplace:.2f}")
print(f"Validation Bigram Perplexity (Laplace): {val_perp_bi_laplace:.2f}")

# K-smoothing (k=0.5)
unigram_probs_k = build_k_smoothing(unigram_counts, 0.5, vocabulary_size)
bigram_probs_k = build_k_smoothing(bigram_counts, 0.5, vocabulary_size, unigram_counts)
val_perp_uni_k = calculate_perplexity(val_tokenized, unigram_probs_k, 1)
val_perp_bi_k = calculate_perplexity(val_tokenized_bigram, bigram_probs_k, 2)
print(f"Validation Unigram Perplexity (k=0.5): {val_perp_uni_k:.2f}")
print(f"Validation Bigram Perplexity (k=0.5): {val_perp_bi_k:.2f}")

print("\n--- Test Set Perplexity Results ---")

# Test set perplexity (optional, can comment out if not needed)
test_perp_uni_nosmooth = calculate_perplexity(test_tokenized, unigram_probs_nosmooth, 1)
test_perp_bi_nosmooth = calculate_perplexity(test_tokenized_bigram, bigram_probs_nosmooth, 2)
test_perp_uni_laplace = calculate_perplexity(test_tokenized, unigram_probs_laplace, 1)
test_perp_bi_laplace = calculate_perplexity(test_tokenized_bigram, bigram_probs_laplace, 2)
test_perp_uni_k = calculate_perplexity(test_tokenized, unigram_probs_k, 1)
test_perp_bi_k = calculate_perplexity(test_tokenized_bigram, bigram_probs_k, 2)
print(f"Test Unigram Perplexity (no smoothing): {test_perp_uni_nosmooth:.2f}")
print(f"Test Bigram Perplexity (no smoothing): {test_perp_bi_nosmooth:.2f}")
print(f"Test Unigram Perplexity (Laplace): {test_perp_uni_laplace:.2f}")
print(f"Test Bigram Perplexity (Laplace): {test_perp_bi_laplace:.2f}")
print(f"Test Unigram Perplexity (k=0.5): {test_perp_uni_k:.2f}")
print(f"Test Bigram Perplexity (k=0.5): {test_perp_bi_k:.2f}")




