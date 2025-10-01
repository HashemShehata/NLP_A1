import nltk
nltk.download('punkt_tab')
import re
from collections import Counter

# --- Tokenization mode switch ---
TOKENIZATION_MODE = 'word'  # 'word' or 'byte'

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
    
def split_by_punct(text):
    tokens = re.split(r'([.!?;:,\'’])', text)
    return [t.strip() for t in tokens if t and not t.isspace()]

def tokenize(text,n,tokenize_strategy="nltk",sentence_lower=True):

    sentences = nltk.sent_tokenize(text)
    start_sentence = ['<s>']
    tokenized_sentences = []

    for sentence in sentences:
        if sentence_lower:
            processed_sentence = sentence.lower()
        else:
            processed_sentence = sentence
        start_padding = max(1, n - 1)
        if tokenize_strategy=="nltk":
            tokens = nltk.word_tokenize(processed_sentence)
        elif tokenize_strategy=="whitespace":
            tokens=processed_sentence.split(" ")
        else:
            tokens=split_by_punct(processed_sentence)
        
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


