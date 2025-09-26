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


# new code thats added - hrishab 
# Function to replace rare words with <unk> and build vocabulary
def replace_rare_words_with_unk(reviews, threshold=1):
    # Count word frequencies (excluding <s> and </s>)
    word_counts = Counter(
        token
        for review in reviews
        for token in nltk.word_tokenize(review.lower())
        if token not in ("<s>", "</s>")
    )
    # Build vocabulary: words with frequency > threshold
    vocab = {word for word, count in word_counts.items() if count > threshold}
    vocab.update({"<s>", "</s>", "<unk>"})

    # Replace rare words with <unk>
    new_reviews = []
    for review in reviews:
        tokens = nltk.word_tokenize(review.lower())
        new_tokens = [
            token if token in vocab else "<unk>" for token in tokens
        ]
        new_reviews.append(" ".join(new_tokens))
    return new_reviews, vocab

train_set = read_file("A1_DATASET/train.txt")
random.shuffle(train_set)

# Replace rare words in train_set with <unk>
train_set, vocab = replace_rare_words_with_unk(train_set, threshold=1)

# Do the same for val_df and test_df using the train vocab
val_df = read_file("A1_DATASET/val.txt")
val_df = [
    " ".join([
        token if token in vocab else "<unk>"
        for token in nltk.word_tokenize(review.lower())
    ])
    for review in val_df
]

test_df = read_file("A1_DATASET/val.txt")
test_df = [
    " ".join([
        token if token in vocab else "<unk>"
        for token in nltk.word_tokenize(review.lower())
    ])
    for review in test_df
]

#approx 102 samples
val_ratio = 0.2  
split_index = int(len(train_set) * val_ratio)
# print (split_index)
# print (train_set[:split_index])
val_df = train_set[:split_index]
train_df = train_set[split_index:]

# test_df = read_file("A1_DATASET/val.txt")
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

sorted_counter = sorted(unigram_counts.items(),key=lambda x:x[1],reverse=True)
with open("output_unigram_counts.txt", "w") as f:
    # for key, value in my_dict.items():
        # f.write(f"{key}: {value}\n")=
    for item in sorted_counter:
        # print (item[0])
        f.write(f"{item[0]}: {item[1]}\n")


# For validation and test get the ngrams only as their prob will be fetched from train
val_bigram_counts = build_ngram(val_df, 2) 
# print (val_bigram_counts)
# val_trigram_counts = build_ngram(val_df, 3) 
# print (val_trigram_counts)
val_unigram_counts = build_ngram(val_df, 1) 
# print (val_unigram_counts)
test_bigram_counts = build_ngram(test_df, 2) 
# print (val_bigram_counts)
# val_trigram_counts = build_ngram(val_df, 3) 
# print (val_trigram_counts)
test_unigram_counts = build_ngram(test_df, 1) 

unigram_probs = build_ngram_probabilities(unigram_counts)
# print (unigram_probs)
bigram_probs = build_ngram_probabilities(bigram_counts,unigram_counts)
# print (bigram_probs)
# trigram_probs = build_ngram_probabilities(trigram_counts,bigram_counts)
# print (trigram_probs)
unigram_probs = build_ngram_laplace_smoothing(unigram_counts,vocabulary_size)
# print (unigram_probs)
bigram_probs = build_ngram_laplace_smoothing(bigram_counts,vocabulary_size,unigram_counts)
# print (bigram_probs)
# trigram_probs = build_ngram_laplace_smoothing(trigram_counts,vocabulary_size,bigram_counts)
# print (trigram_probs)

unigram_probs = build_k_smoothing(unigram_counts,0.5,vocabulary_size)
# print (unigram_probs)
bigram_probs = build_k_smoothing(bigram_counts,0.5,vocabulary_size,unigram_counts)
# print (bigram_probs)
# trigram_probs = build_k_smoothing(trigram_counts,0.5,vocabulary_size,bigram_counts)
# print (trigram_probs)

# print (",..................................\n\n")
# unigram_keys_not_in_train = compare_dicts(unigram_counts,val_unigram_counts)
# print (unigram_keys_not_in_train)
# bigram_keys_not_in_train = compare_dicts(bigram_counts,val_bigram_counts)
# print (bigram_keys_not_in_train)
# sys.exit(0)


