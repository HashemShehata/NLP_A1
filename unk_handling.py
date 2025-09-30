from data_tokenization import tokenize_switch
from ngram_calc import build_ngram
from main import train_df




# First pass: get unigram counts
raw_unigram_counts = build_ngram(train_df, 1)

# Find tokens with freq=1
rare_tokens = set([token for token, count in raw_unigram_counts.items() if count == 2])

def replace_rare_with_unk_tokenized(sentences, rare_tokens, n):
    tokenized_reviews = []
    for review in sentences:
        tokens = tokenize_switch(review, n)
        new_tokens = ["<unk>" if (token,) in rare_tokens else token for token in tokens]
        tokenized_reviews.append(new_tokens)
    return tokenized_reviews

# Replace rare tokens in train_df for unigrams (tokenized)
train_tokenized_unk = replace_rare_with_unk_tokenized(train_df, rare_tokens, 1)

# # not needed only used for testing
# # Function to replace OOV tokens with <unk> in tokenized data
# def replace_oov_with_unk(tokenized_reviews, train_vocab):
#     return [[token if token in train_vocab else '<unk>' for token in tokens] for tokens in tokenized_reviews]
