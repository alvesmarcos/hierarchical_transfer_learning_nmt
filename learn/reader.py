import os
import logging

import utils
from step import Step

logger = logging.getLogger('pipeline')

class Reader(Step):
    command_name = 'reader'
    timestamp = None
    
    def __init__(self, *args, **kwargs):
        logger.info('Reader constructed')
        self.__train_path = kwargs['train_path']
        self.__valid_path = kwargs['valid_path']
        self.__test_path = kwargs['test_path']
        self.__sample = kwargs['sample']
        self.timestamp = args[0]
        self.group = args[1]
    
    def __corpus(self, path, sample=1):
        src_path, tgt_path = path[0], path[1]
        src_sentences = list()
        tgt_sentences = list()
        # transform file to array
        with open(src_path, 'r') as f:
            for line in f:
                src_sentences.append(line[:-1]) # remove `\n`
        with open(tgt_path, 'r') as f:
            for line in f:
                tgt_sentences.append(line[:-1]) # remove `\n`

        src_sentences = src_sentences[:utils.offset(src_sentences, sample)]
        tgt_sentences = tgt_sentences[:utils.offset(tgt_sentences, sample)]
        logger.info(
            f"Source length {len(src_sentences)} and Target length filtered {len(tgt_sentences)}")
        return src_sentences, tgt_sentences

    def routine(self, _input=None):
        train = self.__corpus(self.__train_path, self.__sample)
        logger.info("Train original set")
        valid = self.__corpus(self.__valid_path)
        logger.info("Valid original set")
        test =  self.__corpus(self.__test_path)
        logger.info("Test original set")

        return { 'train': train, 'valid': valid, 'test': test }
