import os
import random
random.seed(42)

from ngram_calc import build_ngram, build_ngram_from_tokenized, build_ngram_probabilities
from data_tokenization import read_file,write_to_file, compare_dicts, tokenize_switch
from perplexity import perplexity
from unk_handling import replace_rare_with_unk_tokenized, set_threshold, get_raw_counts, replace_oov_with_unk
from smoothing import get_k_smoothing, build_k_smoothing
from print_results import  print_stupid_backoff_results, print_no_smoothing_results, print_kn_results
from print_results import print_perplexity_results,print_k_smoothing_results

base_directory = "A1_DATASET/"

### Read the files
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

# Build ngram counts
train_unigram_counts = build_ngram(train_df, 1) 
train_bigram_counts = build_ngram(train_df, 2) 

vocabulary_size = len(set(train_unigram_counts.keys()))
print (vocabulary_size)

write_to_file(train_unigram_counts,"output_unigram_counts.txt")
write_to_file(train_bigram_counts,"output_bigram_counts.txt")

# For validation and test get the ngrams only as their prob will be fetched from train
val_unigram_counts = build_ngram(val_df, 1) 
val_bigram_counts = build_ngram(val_df, 2) 
test_unigram_counts = build_ngram(test_df, 1) 
test_bigram_counts = build_ngram(test_df, 2) 

### Unsmoothed version 
train_unigram_probs = build_ngram_probabilities(train_unigram_counts)
write_to_file(train_unigram_probs,"output_unigram_probs.txt",sort=True)

train_bigram_probs = build_ngram_probabilities(train_bigram_counts,train_unigram_counts)
write_to_file(train_bigram_probs,"output_bigram_probs.txt",sort=True)

print (f"Validation tokens not found in train")
print (len(compare_dicts(train_unigram_counts,val_unigram_counts)))
print (len(compare_dicts(train_bigram_counts,val_bigram_counts)))
print (f"Test tokens not found in train")
print (len(compare_dicts(train_unigram_counts,test_unigram_counts)))
print (len(compare_dicts(train_bigram_counts,test_bigram_counts)))

# No smoothing perplexity results NO UNK
print("NO SMOOTHING RESULTS NO UNK")
print("train results:")
print_no_smoothing_results(train_unigram_counts)
print_no_smoothing_results(train_bigram_counts, train_unigram_counts, None)
print("validation results:")
print_no_smoothing_results(val_unigram_counts, None, train_unigram_probs)
print_no_smoothing_results(val_bigram_counts, None, train_unigram_probs)
print("test results:")
print_no_smoothing_results(test_unigram_counts, None, train_unigram_probs)
print_no_smoothing_results(test_bigram_counts, None, train_unigram_probs)
print()

### Laplace Smoothing NO UNK version 
k= 1
train_unigram_probs_k_smoothing = build_k_smoothing(train_unigram_counts, k,vocabulary_size)
train_bigram_probs_k_smoothing = build_k_smoothing(train_bigram_counts, k, vocabulary_size, train_unigram_counts)

print("LAPLACE SMOOTHING RESULTS NO UNK")
print("train results:")
print_perplexity_results(train_unigram_probs_k_smoothing,train_unigram_counts)
print_perplexity_results(train_bigram_probs_k_smoothing,train_bigram_counts)
print("validation results:")
print_k_smoothing_results(val_unigram_counts,k,vocabulary_size,train_unigram_counts,train_prob_vocab=train_unigram_probs_k_smoothing)
print_k_smoothing_results(val_bigram_counts,k,vocabulary_size,train_bigram_counts,train_unigram_counts,train_bigram_probs_k_smoothing)
print("test results:")
print_k_smoothing_results(test_unigram_counts,k,vocabulary_size,train_unigram_counts,train_prob_vocab= train_unigram_probs_k_smoothing)
print_k_smoothing_results(test_bigram_counts,k,vocabulary_size,train_bigram_counts,train_unigram_counts,train_bigram_probs_k_smoothing)

### K Smoothed version 
print("K Smoothing no UNK version ")
for k in [0.001,0.01,0.02,0.04,0.08,0.1,0.2,0.4,0.8,1]:
    print (f"For value of k ,{k}:")
    unigram_probs = build_k_smoothing(train_unigram_counts,k,vocabulary_size)
    bigram_probs = build_k_smoothing(train_bigram_counts,k,vocabulary_size,train_unigram_counts)
    print (f"For k={k}:")
    print("train results:")
    print_perplexity_results(unigram_probs,train_unigram_counts)
    print_perplexity_results(bigram_probs,train_bigram_counts)
    print("validation results:")
    print_k_smoothing_results(val_unigram_counts,k,vocabulary_size,train_unigram_counts,train_prob_vocab=unigram_probs)
    print_k_smoothing_results(val_bigram_counts,k,vocabulary_size,train_bigram_counts,train_unigram_counts,bigram_probs)
    print("test results:")
    print_k_smoothing_results(test_unigram_counts,k,vocabulary_size,train_unigram_counts,train_prob_vocab= unigram_probs)
    print_k_smoothing_results(test_bigram_counts,k,vocabulary_size,train_bigram_counts,train_unigram_counts,bigram_probs)

### Kneser-Ney Smoothing
print("Kneser-Ney Smoothing no UNK version ")

# Build KN probability function for training data
print("validation results:")
print_kn_results(train_bigram_counts, train_unigram_counts, val_bigram_counts, discount=0.75)
print("test results:")
print_kn_results(train_bigram_counts, train_unigram_counts, test_bigram_counts, discount=0.75)

print("************************EVERYTHING BELOW IS WITH UNK HANDLING********************")
print()
##############################################################################
#                              UNK HANDLING                                  #
##############################################################################

# 1. Determine rare tokens in training (threshold == 1 -> replace frequency <=1)
train_raw_unigram_counts = build_ngram(train_df, 1)
rare_tokens = set_threshold(train_raw_unigram_counts, threshold=2)

# 2. Replace rares with <unk> in training and rebuild counts
train_tokenized_unk = replace_rare_with_unk_tokenized(train_df, rare_tokens, 1)
train_unigram_counts_unk = build_ngram_from_tokenized(train_tokenized_unk, 1)
train_bigram_counts_unk = build_ngram_from_tokenized(train_tokenized_unk, 2)

# # 2.5 Probabilities for <unk> aware training counts
# train_unigram_probs_unk = build_ngram_probabilities(train_unigram_counts_unk)
# train_bigram_probs_unk = build_ngram_probabilities(train_bigram_counts_unk, train_unigram_counts_unk)

# 3. Final training vocab (after UNK replacement)
train_vocab = set(token for (token,) in train_unigram_counts_unk.keys())
vocabulary_size_unk = len(train_vocab)
# 4. Tokenize val/test (unigrams only)
val_tokenized_raw = [tokenize_switch(review, 1) for review in val_df]
test_tokenized_raw = [tokenize_switch(review, 1) for review in test_df]

# 5. Map OOV to <unk>
val_tokenized_unk = replace_oov_with_unk(val_tokenized_raw, train_vocab)
test_tokenized_unk = replace_oov_with_unk(test_tokenized_raw, train_vocab)

# 6. Rebuild val/test UNK-aware counts
val_unigram_counts_unk = build_ngram_from_tokenized(val_tokenized_unk, 1)
val_bigram_counts_unk = build_ngram_from_tokenized(val_tokenized_unk, 2)
test_unigram_counts_unk = build_ngram_from_tokenized(test_tokenized_unk, 1)
test_bigram_counts_unk = build_ngram_from_tokenized(test_tokenized_unk, 2)

# Output unigram counts
sorted_unigram_counter = sorted(train_unigram_counts_unk.items(), key=lambda x: x[1], reverse=True)
with open("output_unigram_counts_with_unk.txt", "w") as f:
    for item in sorted_unigram_counter:
        f.write(f"{item[0]}: {item[1]}\n")

# Output bigram counts
sorted_bigram_counter = sorted(train_bigram_counts_unk.items(), key=lambda x: x[1], reverse=True)
with open("output_bigram_counts_with_unk.txt", "w") as f:
    for item in sorted_bigram_counter:
        f.write(f"{item[0]}: {item[1]}\n")

train_unigram_probs_unk = build_ngram_probabilities(train_unigram_counts_unk)
write_to_file(train_unigram_probs_unk,"output_unigram_probs_unk.txt",sort=True)

train_bigram_probs_unk = build_ngram_probabilities(train_bigram_counts_unk,train_unigram_counts_unk)
write_to_file(train_bigram_probs_unk,"output_bigram_probs_unk.txt",sort=True)

print("NO SMOOTHING RESULTS AFTER UNK HANDLING")
print("train results:")
print_no_smoothing_results(train_unigram_counts_unk)
print_no_smoothing_results(train_bigram_counts_unk, train_unigram_counts_unk, None)
print("validation results:")
print_no_smoothing_results(val_unigram_counts_unk, None, train_unigram_probs_unk)
print_no_smoothing_results(val_bigram_counts_unk, None, train_bigram_probs_unk)
print("test results:")
print_no_smoothing_results(test_unigram_counts_unk, None, train_unigram_probs_unk)
print_no_smoothing_results(test_bigram_counts_unk, None, train_bigram_probs_unk)
print()

print("STUPID BACKOFF RESULTS AFTER UNK HANDLING")
print_stupid_backoff_results(train_bigram_counts_unk, train_unigram_counts_unk, val_bigram_counts_unk, alpha=0.4)
print_stupid_backoff_results(train_bigram_counts_unk, train_unigram_counts_unk, test_bigram_counts_unk, alpha=0.4)
print()

k= 1
train_unigram_probs_k_smoothing_unk = build_k_smoothing(train_unigram_counts_unk, k, vocabulary_size_unk)
train_bigram_probs_k_smoothing_unk = build_k_smoothing(train_bigram_counts_unk, k, vocabulary_size_unk, train_unigram_counts_unk)

print("LAPLACE SMOOTHING RESULTS WITH UNK")
print("train results:")
print_perplexity_results(train_unigram_probs_k_smoothing_unk,train_unigram_counts_unk)
print_perplexity_results(train_bigram_probs_k_smoothing_unk,train_bigram_counts_unk)
print("validation results:")
print_k_smoothing_results(val_unigram_counts_unk,k,vocabulary_size_unk,train_unigram_counts_unk,train_prob_vocab=train_unigram_probs_k_smoothing_unk)
print_k_smoothing_results(val_bigram_counts_unk,k,vocabulary_size_unk,train_bigram_counts_unk,train_unigram_counts_unk,train_bigram_probs_k_smoothing_unk)
print("test results:")
print_k_smoothing_results(test_unigram_counts_unk,k,vocabulary_size_unk,train_unigram_counts_unk,train_prob_vocab= train_unigram_probs_k_smoothing_unk)
print_k_smoothing_results(test_bigram_counts_unk,k,vocabulary_size_unk,train_bigram_counts_unk,train_unigram_counts_unk,train_bigram_probs_k_smoothing_unk)

### K Smoothed version 
print("K Smoothing with UNK version ")
for k in [0.001,0.01,0.02,0.04,0.08,0.1,0.2,0.4,0.8,1]:
    print (f"For value of k ,{k}:")
    train_unigram_probs_unk = build_k_smoothing(train_unigram_counts_unk,k,vocabulary_size_unk)
    train_bigram_probs_unk = build_k_smoothing(train_bigram_counts_unk,k,vocabulary_size_unk,train_unigram_counts_unk)
    print (f"For k={k}:")
    print("train results:")
    print_perplexity_results(train_unigram_probs_unk,train_unigram_counts_unk)
    print_perplexity_results(train_bigram_probs_unk,train_bigram_counts_unk)
    print("validation results:")
    print_k_smoothing_results(val_unigram_counts_unk,k,vocabulary_size_unk,train_unigram_counts_unk,train_prob_vocab=train_unigram_probs_unk)
    print_k_smoothing_results(val_bigram_counts_unk,k,vocabulary_size_unk,train_bigram_counts_unk,train_unigram_counts_unk,train_bigram_probs_unk)
    print("test results:")
    print_k_smoothing_results(test_unigram_counts_unk,k,vocabulary_size_unk,train_unigram_counts_unk,train_prob_vocab= train_unigram_probs_unk)
    print_k_smoothing_results(test_bigram_counts_unk,k,vocabulary_size_unk,train_bigram_counts_unk,train_unigram_counts_unk,train_bigram_probs_unk)


print("Kneser-Ney Smoothing AFTER UNK HANDLING")
print("validation results:")
print_kn_results(train_bigram_counts_unk, train_unigram_counts_unk, val_bigram_counts_unk, discount=0.75)
print("test results:")
print_kn_results(train_bigram_counts_unk, train_unigram_counts_unk, test_bigram_counts_unk, discount=0.75)
print()