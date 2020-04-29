import re

import utils
from step import Step

class Selection(Step):
    command_name = 'selection'
    timestamp = None
    
    def __init__(self, *args, **kwargs):
        self.__max_length = kwargs['max_length']

    def __eng_prefix(self, line):
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

    def __normalize_string(self, s):
        # lowercase, trim and remove non-letter characters 
        s = utils.unicode_to_ascii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    def __get_pairs(self, df):
        # filter by number of words in a sentence
        filter = lambda p: len(p[0].split()) < self.__max_length and len(p[1].split()) < self.__max_length
        return [row for _, row in df.iterrows() if filter(row)]

    def routine(self, _input):
        pairs = list()
        src_lang = list()
        tar_lang = list()
        corpus = self.__get_pairs(_input)

        for source, target in corpus:
            source = self.__normalize_string(self.__eng_prefix(source))
            target = self.__normalize_string(target)
            src_lang.append(source)
            tar_lang.append(target)
        # filter and split corpus on source language, target language
        return src_lang, tar_lang
