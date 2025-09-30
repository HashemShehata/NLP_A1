from data_tokenization import tokenize_switch
from ngram_calc import build_ngram

# First pass: get unigram counts
def get_raw_counts(df):
    ngram = 1
    return build_ngram(df, ngram)


# Find tokens with freq=1
def set_threshold(raw_unigram_counts, threshold=2):
    return set([token for token, count in raw_unigram_counts.items() if count <= threshold])

def replace_rare_with_unk_tokenized(sentences, rare_tokens, n):
    tokenized_reviews = []
    for review in sentences:
        tokens = tokenize_switch(review, n)
        new_tokens = ["<unk>" if (token,) in rare_tokens else token for token in tokens]
        tokenized_reviews.append(new_tokens)
    return tokenized_reviews

def replace_oov_with_unk(tokenized_reviews, train_vocab):
    """Map tokens not in training vocabulary to <unk>.

    Args:
        tokenized_reviews: List[List[str]] token lists.
        train_vocab: Set[str] of tokens retained after UNK replacement in training.
    Returns:
        New tokenized list with OOV tokens replaced by <unk>.
    """
    return [[tok if (tok in train_vocab or tok == '<unk>') else '<unk>' for tok in tokens]
            for tokens in tokenized_reviews]
