# this file is based on the link below (from PyTorch official documentation Example of NMT)
# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

import re
import unicodedata

import pandas as pd

# max word in a sentence per row
MAX_LENGTH = 10

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def eng_prefix(line):
    prefix = { 
        "i'm": "i am", 
        "you're": "you are",
        "it's": "it is",
        "she's": "she is",
        "he's": "he is", 
        "we're": "we are", 
        "they're": "they are"
    }
    _line = list()
    words = [w.lower() for w in line.split()]
    for w in words:
        _line.append(prefix[w] if w in prefix else w)
    return ' '.join(_line)

def normalize_string(s):
    # lowercase, trim and remove non-letter characters 
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def filter_pairs(df):
    # filter by number of words in a sentence
    filter = lambda p: len(p[0].split()) < MAX_LENGTH and len(p[1].split()) < MAX_LENGTH
    return [row for _, row in df.iterrows() if filter(row)]

def read_data(path, reverse=False):
    # read data and ignore header
    df = pd.read_csv(path, sep="\t", header=None)
    df = df.iloc[::-1] if reverse else df
    # filter sentences with max length
    phrases = filter_pairs(df)
    # should be returned
    pairs = list()
    input_lang = list()
    output_lang = list()

    for source, target in phrases:
        source = normalize_string(eng_prefix(source))
        target = normalize_string(target)
        pairs.append((source, target))
        input_lang.append(source)
        output_lang.append(target)

    return input_lang, output_lang, pairs

def write_corpus(path, output):
    input_lang, output_lang, _ = read_data(path)
    with open(output, 'w') as f:
        for source, target in zip(input_lang, output_lang):
            f.write(f'{source}\t{target}\n')
