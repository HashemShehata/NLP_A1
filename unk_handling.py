

def replace_rare_with_unk_tokenized(sentences, rare_tokens, n):
    tokenized_reviews = []
    for review in sentences:
        tokens = tokenize_switch(review, n)
        new_tokens = ["<unk>" if (token,) in rare_tokens else token for token in tokens]
        tokenized_reviews.append(new_tokens)
    return tokenized_reviews