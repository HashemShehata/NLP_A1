import pandas as pd
from collections import defaultdict,Counter
import nltk
nltk.download('punkt_tab')
import random
random.seed(42)
import sys

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


def build_ngram(df, n):
    ngram_counts = Counter()

    for review in df:
        tokenized_sentences = tokenize(review,n)
        # print (tokenized_sentences)
        tokens_count = Counter(tuple(tokenized_sentences[i:i+n]) for i in range(len(tokenized_sentences) - n + 1))
        ngram_counts.update(tokens_count)
    return ngram_counts

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

train_set = read_file("A1_DATASET/train.txt")
random.shuffle(train_set)

#approx 102 samples
val_ratio = 0.2  
split_index = int(len(train_set) * val_ratio)
# print (split_index)
# print (train_set[:split_index])
val_df = train_set[:split_index]
train_df = train_set[split_index:]

test_df = read_file("A1_DATASET/val.txt")
# print (len(val_df))
# print (len(train_df))
# print (len(test_df))

# Example usage
bigram_counts = build_ngram(train_df, 2) 
# print (bigram_counts)
# trigram_counts = build_ngram(train_df, 3) 
# print (trigram_counts)
unigram_counts = build_ngram(train_df, 1) 
# print (unigram_counts)
vocabulary_size = len(set(unigram_counts.keys()))
print (vocabulary_size)

# For validation and test get the ngrams only as their prob will be fetched from train
val_bigram_counts = build_ngram(val_df, 2) 
# print (val_bigram_counts)
# val_trigram_counts = build_ngram(val_df, 3) 
# print (val_trigram_counts)
val_unigram_counts = build_ngram(val_df, 1) 
# print (val_unigram_counts)
test_bigram_counts = build_ngram(test_df, 2) 
test_unigram_counts = build_ngram(test_df, 1) 

# print (",..................................\n\n")
unigram_keys_not_in_train = compare_dicts(unigram_counts,val_unigram_counts)
print(f"Number of unknown words in validation: {len(unigram_keys_not_in_train)}")

# Replace tokens in val_unigram_counts that are in unigram_keys_not_in_train with <UNK>
new_val_unigram_counts = Counter()
unk_count = 0

for ngram, count in val_unigram_counts.items():
    if ngram in unigram_keys_not_in_train:
        # This token is unknown, replace with <UNK>
        new_val_unigram_counts[('<UNK>',)] += count
        unk_count += count
    else:
        # Keep the original token
        new_val_unigram_counts[ngram] += count

# Show the <UNK> count if it exists
if ('<UNK>',) in new_val_unigram_counts:
    print(f"Total <UNK> tokens in validation: {new_val_unigram_counts[('<UNK>',)]}")

# Update val_unigram_counts to use the version with <UNK>
val_unigram_counts = new_val_unigram_counts

# Verify no more unknown words
remaining_unknown = compare_dicts(unigram_counts, val_unigram_counts)

# Also apply UNK replacement to validation bigrams
new_val_bigram_counts = Counter()
bigram_unk_count = 0

# Get the list of unknown words (just the word strings, not tuples)
unknown_words = {ngram[0] for ngram in unigram_keys_not_in_train}

for bigram, count in val_bigram_counts.items():
    # Replace any unknown words in the bigram with <UNK>
    new_bigram = []
    for word in bigram:
        if word in unknown_words:
            new_bigram.append('<UNK>')
            bigram_unk_count += count
        else:
            new_bigram.append(word)
    
    new_bigram_tuple = tuple(new_bigram)
    new_val_bigram_counts[new_bigram_tuple] += count

# Update val_bigram_counts
val_bigram_counts = new_val_bigram_counts

# Add <UNK> token to training vocabulary for probability calculations
print("\nAdding <UNK> token to training vocabulary...")
original_vocab_size = vocabulary_size
print(f"Original training vocab size: {original_vocab_size}")

# Add <UNK> to unigram_counts with count based on unknown words found
if ('<UNK>',) not in unigram_counts:
    # Use the unk_count we calculated earlier (number of unknown token instances)
    unigram_counts[('<UNK>',)] = unk_count

# Also add <UNK> bigram patterns to training bigrams based on validation data
unk_bigram_count = 0

# Find bigrams in validation data that contain unknown words and add equivalent <UNK> patterns
for bigram, count in new_val_bigram_counts.items():
    if '<UNK>' in bigram:
        # This bigram contains <UNK>, so we should add it to training bigrams if not present
        if bigram not in bigram_counts:
            # Use a proportional count based on the validation occurrence
            bigram_counts[bigram] = max(1, count // 10)  # Scale down the count
            unk_bigram_count += 1


# Update vocabulary size
vocabulary_size = len(unigram_counts)
print(f"New training vocab size with <UNK>: {vocabulary_size}")

# Now write the updated unigram counts (with <UNK>) to file
sorted_counter = sorted(unigram_counts.items(),key=lambda x:x[1],reverse=True)
with open("output_unigram_counts.txt", "w") as f:
    for item in sorted_counter:
        f.write(f"{item[0]}: {item[1]}\n")

# Also write the bigram counts to file
sorted_bigram_counter = sorted(bigram_counts.items(),key=lambda x:x[1],reverse=True)
with open("output_bigram_counts.txt", "w") as f:
    for item in sorted_bigram_counter:
        f.write(f"{item[0]}: {item[1]}\n")

unigram_probs_noSmoothing = build_ngram_probabilities(unigram_counts)
# print (unigram_probs)
bigram_probs_noSmoothing = build_ngram_probabilities(bigram_counts,unigram_counts)
# print (bigram_probs)
unigram_probs_laPlace = build_ngram_laplace_smoothing(unigram_counts,vocabulary_size)
# print (unigram_probs)
bigram_probs_laPlace = build_ngram_laplace_smoothing(bigram_counts,vocabulary_size,unigram_counts)
# print (bigram_probs)

unigram_probs_kSmoothing = build_k_smoothing(unigram_counts,0.1,vocabulary_size)
# print (unigram_probs)
bigram_probs_kSmoothing = build_k_smoothing(bigram_counts,0.1,vocabulary_size,unigram_counts)
# print (bigram_probs)
# trigram_probs = build_k_smoothing(trigram_counts,0.5,vocabulary_size,bigram_counts)
# print (trigram_probs)

def calculate_perplexity(text_ngrams, ngram_probs, n=1):
    """Calculate perplexity for a set of n-grams"""
    import math
    
    log_prob_sum = 0
    total_ngrams = 0
    missing_ngrams = 0
    zero_prob_ngrams = 0
    unk_used = 0
    
    for ngram, count in text_ngrams.items():
        if ngram in ngram_probs:
            prob = ngram_probs[ngram]
            if prob > 0:  # Avoid log(0)
                log_prob_sum += count * math.log2(prob)
                total_ngrams += count
            else:
                zero_prob_ngrams += 1
        else:
            # Missing n-gram - treat as UNK
            missing_ngrams += 1
            
            # For missing bigrams, try to use <UNK> patterns
            if n == 2:
                # Try different UNK patterns for missing bigrams
                unk_patterns = [
                    ('<UNK>', ngram[1]),  # First word is UNK
                    (ngram[0], '<UNK>'),  # Second word is UNK  
                    ('<UNK>', '<UNK>')    # Both words are UNK
                ]
                
                prob = 0
                for unk_pattern in unk_patterns:
                    if unk_pattern in ngram_probs:
                        prob = ngram_probs[unk_pattern]
                        break
                
                if prob > 0:
                    log_prob_sum += count * math.log2(prob)
                    total_ngrams += count
                    unk_used += 1
                # If no UNK pattern found, skip this n-gram (contributes 0 to probability)
            
            elif n == 1:
                # For missing unigrams, use <UNK> unigram probability
                unk_unigram = ('<UNK>',)
                if unk_unigram in ngram_probs:
                    prob = ngram_probs[unk_unigram]
                    if prob > 0:
                        log_prob_sum += count * math.log2(prob)
                        total_ngrams += count
                        unk_used += 1
    
    if total_ngrams == 0:
        return float('inf'), missing_ngrams, zero_prob_ngrams, unk_used
    
    # Perplexity = 2^(-1/N * sum(log2(P(w))))
    avg_log_prob = log_prob_sum / total_ngrams
    perplexity = 2 ** (-avg_log_prob)
    
    return perplexity, missing_ngrams, zero_prob_ngrams, unk_used

# Calculate unigram perplexities
print("\nUnigram Perplexities:")
print("-" * 30)

unigram_perp_no, missing_uni_no, zero_uni_no, unk_uni_no = calculate_perplexity(test_unigram_counts, unigram_probs_noSmoothing, 1)
print(f"No Smoothing:     {unigram_perp_no:.2f}")

unigram_perp_lap, missing_uni_lap, zero_uni_lap, unk_uni_lap = calculate_perplexity(test_unigram_counts, unigram_probs_laPlace, 1)
print(f"Laplace Smoothing: {unigram_perp_lap:.2f}")

unigram_perp_k, missing_uni_k, zero_uni_k, unk_uni_k = calculate_perplexity(test_unigram_counts, unigram_probs_kSmoothing, 1)
print(f"K=0.1 Smoothing:   {unigram_perp_k:.2f}")

# Calculate bigram perplexities
print("\nBigram Perplexities:")
print("-" * 30)

bigram_perp_no, missing_bi_no, zero_bi_no, unk_bi_no = calculate_perplexity(test_bigram_counts, bigram_probs_noSmoothing, 2)
print(f"No Smoothing:     {bigram_perp_no:.2f}")

bigram_perp_lap, missing_bi_lap, zero_bi_lap, unk_bi_lap = calculate_perplexity(test_bigram_counts, bigram_probs_laPlace, 2)
print(f"Laplace Smoothing: {bigram_perp_lap:.2f}")

bigram_perp_k, missing_bi_k, zero_bi_k, unk_bi_k = calculate_perplexity(test_bigram_counts, bigram_probs_kSmoothing, 2)
print(f"K=0.1 Smoothing:   {bigram_perp_k:.2f}")

# Show coverage statistics
print(f"\nCoverage Statistics:")
print(f"Missing bigrams in training: {missing_bi_no}")
print(f"UNK patterns used for missing bigrams: {unk_bi_no}")
print(f"Zero probability bigrams: {zero_bi_no}")


# No Smoothing probabilities
sorted_unigram_probs_no = sorted(unigram_probs_noSmoothing.items(), key=lambda x: x[1], reverse=True)
with open("output_unigram_probabilities_no_smoothing.txt", "w") as f:
    f.write("Unigram Probabilities (No Smoothing)\n")
    f.write("====================================\n")
    for ngram, prob in sorted_unigram_probs_no:
        f.write(f"{ngram}: {prob:.6f}\n")

sorted_bigram_probs_no = sorted(bigram_probs_noSmoothing.items(), key=lambda x: x[1], reverse=True)
with open("output_bigram_probabilities_no_smoothing.txt", "w") as f:
    f.write("Bigram Probabilities (No Smoothing)\n")
    f.write("===================================\n")
    for ngram, prob in sorted_bigram_probs_no:
        f.write(f"{ngram}: {prob:.6f}\n")

# Laplace Smoothing probabilities
sorted_unigram_probs_lap = sorted(unigram_probs_laPlace.items(), key=lambda x: x[1], reverse=True)
with open("output_unigram_probabilities_laplace.txt", "w") as f:
    f.write("Unigram Probabilities (Laplace Smoothing)\n")
    f.write("=========================================\n")
    for ngram, prob in sorted_unigram_probs_lap:
        f.write(f"{ngram}: {prob:.6f}\n")

sorted_bigram_probs_lap = sorted(bigram_probs_laPlace.items(), key=lambda x: x[1], reverse=True)
with open("output_bigram_probabilities_laplace.txt", "w") as f:
    f.write("Bigram Probabilities (Laplace Smoothing)\n")
    f.write("========================================\n")
    for ngram, prob in sorted_bigram_probs_lap:
        f.write(f"{ngram}: {prob:.6f}\n")

# K-Smoothing probabilities  
sorted_unigram_probs_k = sorted(unigram_probs_kSmoothing.items(), key=lambda x: x[1], reverse=True)
with open("output_unigram_probabilities_k_smoothing.txt", "w") as f:
    f.write("Unigram Probabilities (K=0.1 Smoothing)\n")
    f.write("=======================================\n")
    for ngram, prob in sorted_unigram_probs_k:
        f.write(f"{ngram}: {prob:.6f}\n")

sorted_bigram_probs_k = sorted(bigram_probs_kSmoothing.items(), key=lambda x: x[1], reverse=True)
with open("output_bigram_probabilities_k_smoothing.txt", "w") as f:
    f.write("Bigram Probabilities (K=0.5 Smoothing)\n")
    f.write("======================================\n")
    for ngram, prob in sorted_bigram_probs_k:
        f.write(f"{ngram}: {prob:.6f}\n")
# sys.exit(0)