import nltk
nltk.download('punkt_tab')
from collections import Counter

# --- Tokenization mode switch ---
TOKENIZATION_MODE = 'byte'  # 'word' or 'byte'

def read_file(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            reviews = [line.strip() for line in f.readlines()]
            return reviews
    except Exception as e:
        print (f"Unable to read the file {filename}: {e}")
        return 
    

def byte_tokenize(text, n):
    # Tokenize text into bytes, return list of byte strings
    text_bytes = text.encode('utf-8')
    # For n-gram, treat each byte as a token (as int or as byte string)
    return [str(b) for b in text_bytes]
    

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

def tokenize_switch(text, n):
    if TOKENIZATION_MODE == 'word':
        return tokenize(text, n)
    elif TOKENIZATION_MODE == 'byte':
        return byte_tokenize(text, n)
    else:
        raise ValueError('Unknown tokenization mode')

def compare_dicts(train_dict,val_dict):
    keys_not_in_train = set(val_dict.keys()) - set(train_dict.keys())
    return keys_not_in_train

def write_to_file(data,filename,sort=True):
    with open(filename, "w", encoding="utf-8") as f:
        if isinstance(data, Counter) or isinstance(data, dict):
            if sort:
                sorted_counter = sorted(data.items(),key=lambda x:x[1],reverse=True)
            else: 
                sorted_counter = data.items()
            for item in sorted_counter:
                f.write(f"{item[0]}: {item[1]}\n")
        else:
            for d in data:
                f.write(f"{d}\n")