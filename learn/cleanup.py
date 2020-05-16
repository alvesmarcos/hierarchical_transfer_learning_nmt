import logging
import re

import utils
from step import Step

logger = logging.getLogger('pipeline')

class CleanUp(Step):
    command_name = 'cleanup'
    timestamp = None
    
    def __init__(self, *args, **kwargs):
        logger.info('CleanUp constructed')
        self.__max_length = kwargs['max_length']

    def __normalize_string(self, s):
        # lowercase, trim and remove non-letter characters 
        s = utils.unicode_to_ascii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    def __get_pairs(self, src_sentences, tgt_sentences):
        src_sentences_filtered = list()
        tgt_sentences_filtered = list()
        # filter by number of words in a sentence
        _filter = lambda p: len(p[0].split()) < self.__max_length and len(p[1].split()) < self.__max_length
        for src, tgt in zip(src_sentences, tgt_sentences):
            if _filter([src, tgt]):
                src_sentences_filtered.append(
                    self.__normalize_string(src)
                )
                tgt_sentences_filtered.append(
                    self.__normalize_string(tgt)
                )
        logger.info(
            f"Source length filtered {len(src_sentences_filtered)} and Target length filtered {len(tgt_sentences_filtered)}")
        return src_sentences_filtered, tgt_sentences_filtered

    def routine(self, _input):
        X_train, y_train = self.__get_pairs(_input['train'][0], _input['train'][1])
        logger.info("Train set")
        X_valid, y_valid = self.__get_pairs(_input['valid'][0], _input['valid'][1])
        logger.info("Valid set")
        X_test, y_test = self.__get_pairs(_input['test'][0], _input['test'][1])
        logger.info("Test set")
        # train/valid/test source and target filtered 
        return (X_train, X_valid, X_test), (y_train, y_valid, y_test)
