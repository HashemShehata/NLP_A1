import os
import random
random.seed(42)

from ngram_calc import build_ngram, build_ngram_probabilities
from data_tokenization import read_file,write_to_file, compare_dicts
from perplexity import perplexity
from unk_handling import replace_rare_with_unk_tokenized, replace_oov_with_unk, rare_tokens
from smoothing import build_kneser_ney_bigram_probs, build_stupid_backoff_bigram_probs, get_k_smoothing, build_k_smoothing
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
unigram_counts = build_ngram(train_df, 1) 
bigram_counts = build_ngram(train_df, 2) 

vocabulary_size = len(set(unigram_counts.keys()))
print (vocabulary_size)

write_to_file(unigram_counts,"output_unigram_counts.txt")
write_to_file(bigram_counts,"output_bigram_counts.txt")

# For validation and test get the ngrams only as their prob will be fetched from train
val_unigram_counts = build_ngram(val_df, 1) 
val_bigram_counts = build_ngram(val_df, 2) 
test_unigram_counts = build_ngram(test_df, 1) 
test_bigram_counts = build_ngram(test_df, 2) 


# Replace rare tokens in train_df for unigrams (tokenized)
train_tokenized_unk = replace_rare_with_unk_tokenized(train_df, rare_tokens, 1)

# Build set of known tokens (training vocab, after <unk> replacement)
train_vocab = set(token for (token,), count in unigram_counts.items())

# # Replace OOV tokens with <unk> in val/test sets
# val_tokenized = replace_oov_with_unk(val_tokenized, train_vocab)
# test_tokenized = replace_oov_with_unk(test_tokenized, train_vocab)


### Unsmoothed version 
unigram_probs = build_ngram_probabilities(unigram_counts)
write_to_file(unigram_probs,"output_unigram_probs.txt",sort=True)

bigram_probs = build_ngram_probabilities(bigram_counts,unigram_counts)
write_to_file(bigram_probs,"output_bigram_probs.txt",sort=True)

print (f"Validation tokens not found in train")
print (len(compare_dicts(unigram_counts,val_unigram_counts)))
print (len(compare_dicts(bigram_counts,val_bigram_counts)))
print (f"Test tokens not found in train")
print (len(compare_dicts(unigram_counts,test_unigram_counts)))
print (len(compare_dicts(bigram_counts,test_bigram_counts)))

val_unigram_probs = build_ngram_probabilities(val_unigram_counts,None,unigram_probs)

val_bigram_probs = build_ngram_probabilities(val_bigram_counts,None,bigram_probs)

print (f"Perpelexity of validation set on unigrams is ",perplexity(val_unigram_probs,val_unigram_counts))
print (f"Perpelexity of validation set on bigrams is ",perplexity(val_bigram_probs,val_bigram_counts))

test_unigram_probs = build_ngram_probabilities(test_unigram_counts,None,unigram_probs)

test_bigram_probs = build_ngram_probabilities(test_bigram_counts,None,bigram_probs)

print (f"Perpelexity of test set on unigrams is ",perplexity(test_unigram_probs,test_unigram_counts))
print (f"Perpelexity of test set on bigrams is ",perplexity(test_bigram_probs,test_bigram_counts))

### Laplace Smoothed version 
k=1
unigram_probs = build_k_smoothing(unigram_counts,k,vocabulary_size)
# write_to_file(unigram_probs,"output_unigram_probs.txt",sort=True)

bigram_probs = build_k_smoothing(bigram_counts,k,vocabulary_size,unigram_counts)
# write_to_file(bigram_probs,"output_bigram_probs.txt",sort=True)

print (f"Validation tokens not found in train")
write_to_file(list(compare_dicts(unigram_counts,val_unigram_counts)),'not_found_unigrams.txt')
write_to_file(list(compare_dicts(bigram_counts,val_bigram_counts)),'not_found_bigrams.txt')

# print (len(compare_dicts(bigram_counts,val_bigram_counts)))


val_unigram_probs = get_k_smoothing(val_unigram_counts,k,vocabulary_size,unigram_counts,train_prob_vocab=unigram_probs)

val_bigram_probs = get_k_smoothing(val_bigram_counts,k,vocabulary_size,bigram_counts,unigram_counts,bigram_probs)

print (f"Perpelexity of validation set on unigrams is ",perplexity(val_unigram_probs,val_unigram_counts))
print (f"Perpelexity of validation set on bigrams is ",perplexity(val_bigram_probs,val_bigram_counts))

test_unigram_probs = get_k_smoothing(test_unigram_counts,k,vocabulary_size,unigram_counts,train_prob_vocab=unigram_probs)

test_bigram_probs = get_k_smoothing(test_bigram_counts,k,vocabulary_size,bigram_counts,unigram_counts,bigram_probs)

print (f"Perpelexity of test set on unigrams is ",perplexity(test_unigram_probs,test_unigram_counts))
print (f"Perpelexity of test set on bigrams is ",perplexity(test_bigram_probs,test_bigram_counts))


### K Smoothed version 

for k in [0.001,0.01,0.02,0.04,0.08,0.1,0.2,0.4,0.8,1]:
    print (f"For value of k ,{k}:")
    unigram_probs = build_k_smoothing(unigram_counts,k,vocabulary_size)
    write_to_file(unigram_probs,"output_unigram_probs.txt",sort=True)

    bigram_probs = build_k_smoothing(bigram_counts,k,vocabulary_size,unigram_counts)
    write_to_file(bigram_probs,"output_bigram_probs.txt",sort=True)

    val_unigram_probs = get_k_smoothing(val_unigram_counts,k,vocabulary_size,unigram_counts,train_prob_vocab=unigram_probs)

    val_bigram_probs = get_k_smoothing(val_bigram_counts,k,vocabulary_size,bigram_counts,unigram_counts,bigram_probs)

    print (f"Perpelexity of validation set on unigrams is ",perplexity(val_unigram_probs,val_unigram_counts))
    print (f"Perpelexity of validation set on bigrams is ",perplexity(val_bigram_probs,val_bigram_counts))

    test_unigram_probs = get_k_smoothing(test_unigram_counts,k,vocabulary_size,unigram_counts,train_prob_vocab=unigram_probs)

    test_bigram_probs = get_k_smoothing(test_bigram_counts,k,vocabulary_size,bigram_counts,unigram_counts,bigram_probs)

    print (f"Perpelexity of test set on unigrams is ",perplexity(test_unigram_probs,test_unigram_counts))
    print (f"Perpelexity of test set on bigrams is ",perplexity(test_bigram_probs,test_bigram_counts))


### Kneser-Ney Smoothing
print("\n" + "="*50)
print("KNESER-NEY SMOOTHING")
print("="*50)

# Build KN probability function for training data
kn_prob_func = build_kneser_ney_bigram_probs(bigram_counts, unigram_counts, discount=0.75)

# Calculate probabilities for validation set
val_bigram_probs_kn = {}
for bigram in val_bigram_counts.keys():
    w1, w2 = bigram
    val_bigram_probs_kn[bigram] = kn_prob_func(w1, w2)

print(f"Perplexity of validation set on bigrams (KN): {perplexity(val_bigram_probs_kn, val_bigram_counts)}")

# Calculate probabilities for test set
test_bigram_probs_kn = {}
for bigram in test_bigram_counts.keys():
    w1, w2 = bigram
    test_bigram_probs_kn[bigram] = kn_prob_func(w1, w2)

print(f"Perplexity of test set on bigrams (KN): {perplexity(test_bigram_probs_kn, test_bigram_counts)}")

# Try different discount values
print("\nKneser-Ney with different discount values:")
for discount in [0.5, 0.65, 0.75, 0.85, 0.95]:
    kn_prob_func = build_kneser_ney_bigram_probs(bigram_counts, unigram_counts, discount=discount)
    
    val_bigram_probs_kn = {bigram: kn_prob_func(bigram[0], bigram[1]) for bigram in val_bigram_counts.keys()}
    test_bigram_probs_kn = {bigram: kn_prob_func(bigram[0], bigram[1]) for bigram in test_bigram_counts.keys()}
    
    print(f"    Val Perplexity: {perplexity(val_bigram_probs_kn, val_bigram_counts)}")
    print(f"    Test Perplexity: {perplexity(test_bigram_probs_kn, test_bigram_counts)}")


### Stupid Backoff Smoothing
print("\n" + "="*50)
print("STUPID BACKOFF SMOOTHING")
print("="*50)

# Build SB probability function for training data
sb_prob_func = build_stupid_backoff_bigram_probs(bigram_counts, unigram_counts, alpha=0.4)

# Calculate probabilities for validation set
val_bigram_probs_sb = {}
for bigram in val_bigram_counts.keys():
    w1, w2 = bigram
    val_bigram_probs_sb[bigram] = sb_prob_func(w1, w2)

print(f"Perplexity of validation set on bigrams (SB): {perplexity(val_bigram_probs_sb, val_bigram_counts)}")

# Calculate probabilities for test set
test_bigram_probs_sb = {}
for bigram in test_bigram_counts.keys():
    w1, w2 = bigram
    test_bigram_probs_sb[bigram] = sb_prob_func(w1, w2)

print(f"Perplexity of test set on bigrams (SB): {perplexity(test_bigram_probs_sb, test_bigram_counts)}")

# Try different alpha values
print("\nStupid Backoff with different alpha values:")
for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    sb_prob_func = build_stupid_backoff_bigram_probs(bigram_counts, unigram_counts, alpha=alpha)
    
    val_bigram_probs_sb = {bigram: sb_prob_func(bigram[0], bigram[1]) for bigram in val_bigram_counts.keys()}
    test_bigram_probs_sb = {bigram: sb_prob_func(bigram[0], bigram[1]) for bigram in test_bigram_counts.keys()}
    
    print(f"    Val Perplexity: {perplexity(val_bigram_probs_sb, val_bigram_counts)}")
    print(f"    Test Perplexity: {perplexity(test_bigram_probs_sb, test_bigram_counts)}")
