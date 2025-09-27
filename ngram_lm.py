#!/usr/bin/env python3
# N-gram Language Model (Unigram & Bigram) with Smoothing and Perplexity
# -----------------------------------------------------------------------------
# Usage examples:
#
# # Train models and evaluate on validation set
# python ngram_lm.py --train path/to/train.txt --valid path/to/validation.txt \
#   --unk-threshold 1 --k 1.0 --lower --report
#
# # Evaluate multiple k values (grid) for Laplace/add-k smoothing
# python ngram_lm.py --train train.txt --valid validation.txt --k-grid 0.1,0.5,1,2,5
#
# Notes:
# - Input files are expected to have one review per line, already tokenized by space.
# - This script handles <UNK> replacement using a frequency threshold on the training set.
# - Smoothing options: add-k (Laplace when k=1). (Second smoother: Stupid Backoff.)
# - Outputs perplexity and sample probabilities for sanity checks.
# -----------------------------------------------------------------------------

import argparse
import math
from collections import Counter
from typing import List, Tuple, Set

import nltk

# Ensure punkt is available for sentence tokenization
nltk.download('punkt')

# Read and preprocess the data: tokenization, lowercasing, add <s> and </s>
def read_reviews_and_sentences(path, lower=False):
    BOS = "<s>"   # Special token for Beginning Of Sentence
    EOS = "</s>"  # Special token for End Of Sentence
    reviews = []  # List to store all reviews; each review is a list of sentences (each as a list of tokens)

    with open(path, "r", encoding="utf-8") as f:
        for review in f:  # Iterate over each line in the file; each line is a review
            review = review.strip()  # Remove leading/trailing whitespace from the review
            if not review:
                continue  # Skip empty lines

            sentences = []  # List to store sentences for the current review

            # Split the review into sentences using NLTK's sentence tokenizer
            for sentence in nltk.sent_tokenize(review):
                tokens = sentence.split()  # Split the sentence into words/tokens by whitespace
                if lower:
                    tokens = [tok.lower() for tok in tokens]  # Lowercase all tokens if requested
                tokens = [BOS] + tokens + [EOS]  # Add sentence boundary tokens
                sentences.append(tokens)  # Add processed sentence (list of tokens) to sentences list

            reviews.append(sentences)  # Add the current review (list of sentences) to reviews list

    # reviews: List of reviews; each review is a list of sentences; each sentence is a list of tokens (with BOS/EOS)
    return reviews


def build_vocab_and_replace_unk(reviews, threshold=1):
    # Count word frequencies across all sentences in all reviews (excluding BOS and EOS), These are special boundary tokens added to every sentence, not actual words from the reviews.
    # It iterates through every token in every sentence in every review.
    # For each token, it increments its count.
    word_counts = Counter(
        token
        for review in reviews
        for sentence in review
        for token in sentence
        # Exclude special tokens from counting as they are not real words like bos and eos
        if token not in ("<s>", "</s>")
    )

    # Build vocabulary: words with frequency > threshold, the threshold default is 1
    # This creates a set of words that appear more than 'threshold' times in the training data.
    vocab = {word for word, count in word_counts.items() if count > threshold}
    # It also ensures that special tokens <s>, </s>, and <unk> are always included in the vocabulary.
    vocab.update({"<s>", "</s>", "<unk>"}) 

    # Replace rare words with <unk>
    # the new_reviews list will store the processed reviews with rare words replaced by <unk>.
    new_reviews = []

    # Iterate through each review
    for review in reviews:
        new_review = []
        # Iterate through each sentence in the review
        for sentence in review:
            new_sentence = [
                token if token in vocab else "<unk>" for token in sentence
            ]
            new_review.append(new_sentence)
        new_reviews.append(new_review)

    return vocab, new_reviews


def count_ngrams(reviews):
    unigram_counts = Counter()  # Counts for individual tokens
    bigram_counts = Counter()   # Counts for pairs of consecutive tokens

    for review in reviews:
        for sentence in review:
            # Count unigrams
            for token in sentence:
                unigram_counts[token] += 1
            # Count bigrams
            for i in range(len(sentence) - 1):
                bigram = (sentence[i], sentence[i + 1])
                bigram_counts[bigram] += 1

    return unigram_counts, bigram_counts


def compute_unigram_probs(unigram_counts):
    total_tokens = sum(unigram_counts.values())
    unigram_probs = {}
    for token, count in unigram_counts.items():
        unigram_probs[token] = count / total_tokens
    return unigram_probs

def compute_bigram_probs(bigram_counts, unigram_counts):
    bigram_probs = {}
    for (w1, w2), count in bigram_counts.items():
        bigram_probs[(w1, w2)] = count / unigram_counts[w1]
    return bigram_probs


#DEBUGGING / TESTING

# Read and preprocess reviews
reviews = read_reviews_and_sentences("test.txt", lower=True)

# Build vocabulary and replace rare words
vocab, new_reviews = build_vocab_and_replace_unk(reviews, threshold=1)

# Count unigrams and bigrams

unigram_counts, bigram_counts = count_ngrams(new_reviews)


# # print("Count of <unk>:", unigram_counts["<unk>"])
# # print("Total tokens:", sum(unigram_counts.values()))

# # Print the number of unique unigrams and bigrams
# print("Number of unique unigrams:", len(unigram_counts))
# print("Number of unique bigrams:", len(bigram_counts))




# # Print all unigram probabilities as a dictionary
# print("\nAll unigram probabilities:")
# print(compute_unigram_probs(unigram_counts))

# # Print all bigram probabilities as a dictionary
# print("\nAll bigram probabilities:")
# print(compute_bigram_probs(bigram_counts, unigram_counts))

for review in new_reviews:
    for sentence in review:
        print(sentence)

