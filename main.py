import os
import random
random.seed(42)

from ngram_calc import build_ngram, build_ngram_probabilities
from data_tokenization import read_file,write_to_file, compare_dicts
from perplexity import perplexity
from smoothing import get_k_smoothing, build_k_smoothing
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

##