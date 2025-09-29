import pandas as pd
from collections import defaultdict,Counter
import nltk
nltk.download('punkt_tab')
import random
random.seed(42)
import sys
import math
import os

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


def build_ngram(df, n, is_training=False, unk_threshold=1, training_vocab=None):
    ngram_counts = Counter()

    for review in df:
        tokenized_sentences = tokenize(review,n)
        
        # For validation/test: replace unknown tokens with <UNK>
        if not is_training and training_vocab is not None:
            tokenized_sentences = replace_unknown_tokens(tokenized_sentences, training_vocab)
        
        # print (tokenized_sentences)
        tokens_count = Counter(tuple(tokenized_sentences[i:i+n]) for i in range(len(tokenized_sentences) - n + 1))
        ngram_counts.update(tokens_count)
    
    # Post-process: replace low-frequency n-grams with <UNK> versions (only for training)
    if is_training:
        ngram_counts = replace_low_freq_with_unk(ngram_counts, n, unk_threshold)
    
    return ngram_counts

def replace_low_freq_with_unk(ngram_counts, n, unk_threshold):
    """Replace low-frequency n-grams with <UNK> versions"""
    updated_counts = Counter()
    
    for ngram, count in ngram_counts.items():
        if count <= unk_threshold:
            # Replace low-frequency n-grams with <UNK> versions
            unk_ngram = tuple('<UNK>' if token not in ['<s>', '</s>'] else token for token in ngram)
            updated_counts[unk_ngram] += count
        else:
            # Keep high-frequency n-grams as they are
            updated_counts[ngram] = count
    
    return updated_counts

def replace_unknown_tokens(tokenized_sentences, training_vocab):
    """Replace tokens not in training vocabulary with <UNK>"""
    updated_tokens = []
    for token in tokenized_sentences:
        if token in ['<s>', '</s>']:
            # Keep special tokens as they are
            updated_tokens.append(token)
        elif (token,) in training_vocab:
            # Keep known tokens
            updated_tokens.append(token)
        else:
            # Replace unknown tokens with <UNK>
            updated_tokens.append('<UNK>')
    return updated_tokens

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

def compare_dicts(train_dict,val_dict):
    keys_not_in_train = set(val_dict.keys()) - set(train_dict.keys())
    return keys_not_in_train

def calculate_perplexity(test_ngram_counts, ngram_probs, training_counts=None, context_counts=None, vocab_size=None, smoothing_type="none", k=0.5):
    log_prob_sum = 0
    total_ngrams = 0
    
    for ngram, count in test_ngram_counts.items():
        # Try to get probability from pre-calculated probabilities
        prob = ngram_probs.get(ngram, 0)
        
        # If probability is 0 (unseen n-gram), calculate smoothed probability
        if prob == 0 and smoothing_type != "none":
            if smoothing_type == "laplace":
                # Laplace smoothing: (0 + 1) / (context_count + vocab_size)
                context = ngram[:-1] if len(ngram) > 1 else ()
                if context_counts is not None:
                    context_count = context_counts.get(context, 0)
                else:
                    context_count = sum(training_counts.values()) if training_counts else 1
                prob = 1 / (context_count + vocab_size)
                
            elif smoothing_type == "k_smoothing":
                # k-smoothing: (0 + k) / (context_count + k*vocab_size)  
                context = ngram[:-1] if len(ngram) > 1 else ()
                if context_counts is not None:
                    context_count = context_counts.get(context, 0)
                else:
                    context_count = sum(training_counts.values()) if training_counts else 1
                prob = k / (context_count + k * vocab_size)
        
        if prob > 0:
            log_prob_sum += count * math.log2(prob)
        else:
            # Handle zero probability 
            if smoothing_type == "none":
                # For no smoothing, zero probabilities are expected - use small epsilon
                log_prob_sum += count * math.log2(1e-10)
            else:
                # For smoothing methods, this shouldn't happen - show warning
                print(f"Warning: Zero probability for n-gram {ngram} with {smoothing_type} smoothing")
                log_prob_sum += count * math.log2(1e-10)
        
        total_ngrams += count
    
    # Calculate perplexity: 2^(-1/N * sum(log2(p)))
    if total_ngrams == 0:
        return float('inf')
    
    avg_log_prob = log_prob_sum / total_ngrams
    perplexity = 2 ** (-avg_log_prob)
    
    return perplexity

def write_to_file(data,filename):
    with open(filename, "w", encoding="utf-8") as f:
        if isinstance(data, Counter):
            sorted_counter = sorted(data.items(),key=lambda x:x[1],reverse=True)
            for item in sorted_counter:
                f.write(f"{item[0]}: {item[1]}\n")
        else:
            for d in data:
                f.write(d + "\n")

base_directory = "A1_DATASET/"
if not os.path.exists(f"{base_directory}dev_split_20.txt"):
    train_set = read_file(f"{base_directory}train.txt")
    random.shuffle(train_set)

    #approx 102 samples
    val_ratio = 0.2  
    split_index = int(len(train_set) * val_ratio)
    val_df = train_set[:split_index]
    train_df = train_set[split_index:]

    write_to_file(train_df,f"{base_directory}train_split_80.txt")
    write_to_file(val_df,f"{base_directory}dev_split_20.txt")

else:
    print ("Reading directly")
    train_df = read_file(f"{base_directory}train_split_80.txt")
    val_df = read_file(f"{base_directory}dev_split_20.txt")
    
test_df = read_file(f"{base_directory}val.txt")

# Example usage
unigram_counts = build_ngram(train_df, 1, is_training=True, unk_threshold=1) 
bigram_counts = build_ngram(train_df, 2, is_training=False, training_vocab=unigram_counts) 

vocabulary_size = len(set(unigram_counts.keys()))

write_to_file(unigram_counts,"output_unigram_counts.txt")
write_to_file(bigram_counts,"output_bigram_counts.txt")

sys.exit(0)

# For validation and test get the ngrams only as their prob will be fetched from train
val_bigram_counts = build_ngram(val_df, 2, is_training=False, training_vocab=unigram_counts) 
# print (val_bigram_counts)
val_unigram_counts = build_ngram(val_df, 1, is_training=False, training_vocab=unigram_counts) 
# print (val_unigram_counts)
test_bigram_counts = build_ngram(test_df, 2, is_training=False, training_vocab=unigram_counts) 
test_unigram_counts = build_ngram(test_df, 1, is_training=False, training_vocab=unigram_counts) 

unigram_probs = build_ngram_probabilities(unigram_counts)
# print (unigram_probs)
bigram_probs = build_ngram_probabilities(bigram_counts,unigram_counts)
# print (bigram_probs)
unigram_probs = build_ngram_laplace_smoothing(unigram_counts,vocabulary_size)
# print (unigram_probs)
bigram_probs = build_ngram_laplace_smoothing(bigram_counts,vocabulary_size,unigram_counts)
# print (bigram_probs)

unigram_probs = build_k_smoothing(unigram_counts,0.5,vocabulary_size)
# print (unigram_probs)
bigram_probs = build_k_smoothing(bigram_counts,0.5,vocabulary_size,unigram_counts)
# print (bigram_probs)


# Unigram probabilities with different smoothing methods
unigram_probs_no_smooth = build_ngram_probabilities(unigram_counts)
unigram_probs_laplace = build_ngram_laplace_smoothing(unigram_counts, vocabulary_size)
unigram_probs_k = build_k_smoothing(unigram_counts, 0.5, vocabulary_size)
bigram_probs_no_smooth = build_ngram_probabilities(bigram_counts, unigram_counts)
bigram_probs_laplace = build_ngram_laplace_smoothing(bigram_counts, vocabulary_size, unigram_counts)
bigram_probs_k = build_k_smoothing(bigram_counts, 0.5, vocabulary_size, unigram_counts)


# print (",..................................\n\n")
unigram_keys_not_in_train = compare_dicts(unigram_counts,test_unigram_counts)
print (len(unigram_keys_not_in_train))

# Calculate and print perplexities for val, train, and test sets

# Unigram perplexities
val_unigram_perplexity = calculate_perplexity(
    val_unigram_counts, unigram_probs_no_smooth, training_counts=unigram_counts,
    context_counts=None, vocab_size=vocabulary_size, smoothing_type="none"
)
train_unigram_perplexity = calculate_perplexity(
    unigram_counts, unigram_probs_no_smooth, training_counts=unigram_counts,
    context_counts=None, vocab_size=vocabulary_size, smoothing_type="none"
)
test_unigram_perplexity = calculate_perplexity(
    test_unigram_counts, unigram_probs_no_smooth, training_counts=unigram_counts,
    context_counts=None, vocab_size=vocabulary_size, smoothing_type="none"
)

print()
print("Unigram NO SMOOTH")
print(f"Unigram Perplexity (val) no smooth: {val_unigram_perplexity}")
print(f"Unigram Perplexity (train) no smooth: {train_unigram_perplexity}")
print(f"Unigram Perplexity (test) no smooth: {test_unigram_perplexity}")

# Bigram perplexities
val_bigram_perplexity = calculate_perplexity(
    val_bigram_counts, bigram_probs_no_smooth, training_counts=bigram_counts,
    context_counts=unigram_counts, vocab_size=vocabulary_size, smoothing_type="none"
)
train_bigram_perplexity = calculate_perplexity(
    bigram_counts, bigram_probs_no_smooth, training_counts=bigram_counts,
    context_counts=unigram_counts, vocab_size=vocabulary_size, smoothing_type="none"
)
test_bigram_perplexity = calculate_perplexity(
    test_bigram_counts, bigram_probs_no_smooth, training_counts=bigram_counts,
    context_counts=unigram_counts, vocab_size=vocabulary_size, smoothing_type="none"
)

print()
print("Bigram NO SMOOTH")
print(f"Bigram Perplexity (val) no smooth: {val_bigram_perplexity}")
print(f"Bigram Perplexity (train) no smooth: {train_bigram_perplexity}")
print(f"Bigram Perplexity (test) no smooth: {test_bigram_perplexity}")

# Unigram perplexities with Laplace smoothing
val_unigram_perplexity_laplace = calculate_perplexity(
    val_unigram_counts, unigram_probs_laplace, training_counts=unigram_counts,
    context_counts=None, vocab_size=vocabulary_size, smoothing_type="laplace"
)
train_unigram_perplexity_laplace = calculate_perplexity(
    unigram_counts, unigram_probs_laplace, training_counts=unigram_counts,
    context_counts=None, vocab_size=vocabulary_size, smoothing_type="laplace"
)
test_unigram_perplexity_laplace = calculate_perplexity(
    test_unigram_counts, unigram_probs_laplace, training_counts=unigram_counts,
    context_counts=None, vocab_size=vocabulary_size, smoothing_type="laplace"
)

print()
print("Unigram LAPLACE")
print(f"Unigram Perplexity (val) Laplace: {val_unigram_perplexity_laplace}")
print(f"Unigram Perplexity (train) Laplace: {train_unigram_perplexity_laplace}")
print(f"Unigram Perplexity (test) Laplace: {test_unigram_perplexity_laplace}")
# Unigram perplexities with k-smoothing
val_unigram_perplexity_k = calculate_perplexity(
    val_unigram_counts, unigram_probs_k, training_counts=unigram_counts,
    context_counts=None, vocab_size=vocabulary_size, smoothing_type="k_smoothing", k=0.5
)
train_unigram_perplexity_k = calculate_perplexity(
    unigram_counts, unigram_probs_k, training_counts=unigram_counts,
    context_counts=None, vocab_size=vocabulary_size, smoothing_type="k_smoothing", k=0.5
)
test_unigram_perplexity_k = calculate_perplexity(
    test_unigram_counts, unigram_probs_k, training_counts=unigram_counts,
    context_counts=None, vocab_size=vocabulary_size, smoothing_type="k_smoothing", k=0.5
)

print()
print("Unigram K-SMOOTHING")
print(f"Unigram Perplexity (val) k-smoothing: {val_unigram_perplexity_k}")
print(f"Unigram Perplexity (train) k-smoothing: {train_unigram_perplexity_k}")
print(f"Unigram Perplexity (test) k-smoothing: {test_unigram_perplexity_k}")
# Bigram perplexities with Laplace smoothing
val_bigram_perplexity_laplace = calculate_perplexity(
    val_bigram_counts, bigram_probs_laplace, training_counts=bigram_counts,
    context_counts=unigram_counts, vocab_size=vocabulary_size, smoothing_type="laplace"
)
train_bigram_perplexity_laplace = calculate_perplexity(
    bigram_counts, bigram_probs_laplace, training_counts=bigram_counts,
    context_counts=unigram_counts, vocab_size=vocabulary_size, smoothing_type="laplace"
)
test_bigram_perplexity_laplace = calculate_perplexity(
    test_bigram_counts, bigram_probs_laplace, training_counts=bigram_counts,
    context_counts=unigram_counts, vocab_size=vocabulary_size, smoothing_type="laplace"
)

print()
print("Bigram LAPLACE")
print(f"Bigram Perplexity (val) Laplace: {val_bigram_perplexity_laplace}")
print(f"Bigram Perplexity (train) Laplace: {train_bigram_perplexity_laplace}")
print(f"Bigram Perplexity (test) Laplace: {test_bigram_perplexity_laplace}")

# Bigram perplexities with k-smoothing
val_bigram_perplexity_k = calculate_perplexity(
    val_bigram_counts, bigram_probs_k, training_counts=bigram_counts,
    context_counts=unigram_counts, vocab_size=vocabulary_size, smoothing_type="k_smoothing", k=0.5
)
train_bigram_perplexity_k = calculate_perplexity(
    bigram_counts, bigram_probs_k, training_counts=bigram_counts,
    context_counts=unigram_counts, vocab_size=vocabulary_size, smoothing_type="k_smoothing", k=0.5
)
test_bigram_perplexity_k = calculate_perplexity(
    test_bigram_counts, bigram_probs_k, training_counts=bigram_counts,
    context_counts=unigram_counts, vocab_size=vocabulary_size, smoothing_type="k_smoothing", k=0.5
)

print()
print("Bigram K-SMOOTHING")
print(f"Bigram Perplexity (val) k-smoothing: {val_bigram_perplexity_k}")
print(f"Bigram Perplexity (train) k-smoothing: {train_bigram_perplexity_k}")
print(f"Bigram Perplexity (test) k-smoothing: {test_bigram_perplexity_k}")

# sys.exit(0)